from concurrent.futures import thread
import random
from flask import Flask
from waitress import serve
import requests
import pickle
import os
from collections import defaultdict
from os.path import join, dirname
from dotenv import load_dotenv
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import random
import dotenv

app = Flask(__name__)
CORS(app)

# dotenv_path = join(dirname(__file__), '.env')
# load_dotenv(dotenv_path)
# PA_TOKEN = dotenv.get_key(dotenv_path, 'PA_TOKEN')

PA_TOKEN = os.environ.get("PA_TOKEN")

headers = {
    "Authorization": f"Bearer {PA_TOKEN}",
}

column_headers = ["Username", "CreatedAt", "AvatarUrl", "Id", "Contributions", "JavaScript", "Python", "Java", "C#", "PHP", "TypeScript", "Ruby", "C++", "C", "Swift", "Go", "Shell", "Kotlin", "Rust", "PowerShell", "Objective-C", "R", "MATLAB", "Dart", "Vue", "Assembly", "Sass", "CSS", "HTML", "Pascal", "Racket", "Zig", "Other"]
tcols = [x + '-T' for x in column_headers[3:]]
knownLangs = set(column_headers)

gql_query = """
query GetUser($username: String!) {
    user: user(login: $username) { # my username
        login
        id
        createdAt
        avatarUrl
        contributionsCollection {
            contributionCalendar {
                totalContributions
            }
        }
        pinnedItems(first: 6, types: REPOSITORY) {
            nodes {
                ... on Repository {
                    name
                    id
                    languages(first: 10) {
                        edges {
                            size
                        }
                        nodes {
                            name
                        }
                    }
                }
            }
        }
    }
}
"""

with open("./data/kmeansmodel.pkl", "rb") as f:
    model = pickle.load(f)

meanAndStd = pd.read_csv("./data/meanAndStd.csv")
training_data = pd.read_csv("./data/clustered_data.csv", index_col=0)



@app.route('/')
def hello_world():
  return 'Hello world!'


def get_user(username):
    data = {
      'query': gql_query,
      'variables': {
        "username": username
      }
    }
    
    user_csv = ""
    response = requests.post("https://api.github.com/graphql", headers=headers, json=data)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        result = response.json()
    
        try:
            login = result['data']['user']['login']
            createdAt = str(result['data']['user']['createdAt'])
            avatarUrl = result['data']['user']['avatarUrl']
            id = str(random.randint(1, 2))
            contributions = str(result['data']['user']['contributionsCollection']['contributionCalendar']['totalContributions'])

        except:
            return {"success":False, "message":"Invalid user"}

        langs = defaultdict(int)
        num_bytes = 0
        pinned_projects = result['data']['user']['pinnedItems']['nodes']

        for i in range(len(pinned_projects)): # all pinned projects
            for j in range(len(pinned_projects[i]['languages']['nodes'])): # all languages in project
                lang = pinned_projects[i]['languages']['nodes'][j]['name']
                b = pinned_projects[i]['languages']['edges'][j]['size']
                num_bytes += b
                if lang in knownLangs:
                    langs[lang] += b
                else:
                    langs["Other"] += b

        if num_bytes == 0:
            return {"success":False, "message":"User has no pinned repositories..."}

        user_csv = f"{login},{createdAt},{avatarUrl},{id},{contributions}"
        for x in column_headers[5:]:
            user_csv += "," + str(langs[x])
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        raise ValueError

    with open("./data/user.csv", "w") as f:
        f.write(f"{','.join(column_headers)}\n{user_csv}\n")
    return {"success":True}


def standardize(data):
    # convert to proper format
    col = ["Id", "Contributions", "JavaScript", "Python", "Java", "C#", "PHP", "TypeScript", "Ruby", "C++", "C", "Swift", "Go", "Shell", "Kotlin", "Rust", "PowerShell", "Objective-C", "R", "MATLAB", "Dart", "Vue", "Assembly", "Sass", "CSS", "HTML", "Pascal", "Racket", "Zig", "Other"]
    def turn_to_percent(X, columns):
        X[columns] = X[columns].div(X[columns].sum(axis=1), axis=0)
        return X

    # Create a FunctionTransformer using the defined function and pass the subset_columns argument
    transformer = FunctionTransformer(turn_to_percent, validate=False, kw_args={'columns': col[2:]})

    # Apply the transformation to your dataset
    data = transformer.transform(data)

    for lang in meanAndStd:
        data[lang + '-T'] = (data[lang] - meanAndStd[lang][0]) / meanAndStd[lang][1]
    
    # prediction
    predicted_cluster = model.predict(data[tcols])[0]

    condition = training_data['cluster'] == predicted_cluster
    cluster = training_data[condition]
    picked = cluster.sample(n=min(len(cluster), 20), random_state=0)
    ret = []

    for username, row in picked.iterrows():
        curr = {}
        curr['username'] = username
        curr['createdAt'] = row['CreatedAt']
        curr['avatarUrl'] = row['AvatarUrl']
        curr['id'] = row['Id']
        curr['contributions'] = row['Contributions']
        curr['languages'] = {}
        for lang in column_headers[5:]:
            if row[lang] > 0:
                curr['languages'][lang] = row[lang]
        ret.append(curr)
    return ret


@app.route('/find_matches/<username>')
def find_matches(username):
    global model, meanAndStd

    response = get_user(username)
    if not response["success"]:
        return response
    
    users = pd.read_csv("./data/user.csv", index_col=0)

    # convert to proper format
    col = ["Id", "Contributions", "JavaScript", "Python", "Java", "C#", "PHP", "TypeScript", "Ruby", "C++", "C", "Swift", "Go", "Shell", "Kotlin", "Rust", "PowerShell", "Objective-C", "R", "MATLAB", "Dart", "Vue", "Assembly", "Sass", "CSS", "HTML", "Pascal", "Racket", "Zig", "Other"]
    def turn_to_percent(X, columns):
        X[columns] = X[columns].div(X[columns].sum(axis=1), axis=0)
        return X

    # Create a FunctionTransformer using the defined function and pass the subset_columns argument
    transformer = FunctionTransformer(turn_to_percent, validate=False, kw_args={'columns': col[2:]})

    # Apply the transformation to your dataset
    users = transformer.transform(users)

    user = users.iloc[0]
    print(user)


    standardized_user = {
        "username": username,
        "createdAt": user["CreatedAt"],
        "avatarUrl": user["AvatarUrl"],
        "id": str(user["Id"]),
        "contributions": str(user["Contributions"]),
        "languages": {}
    }

    for lang in column_headers[5:]:
        if user[lang] > 0:
            standardized_user["languages"][lang] = str(user[lang])
    print(standardized_user)

    return {
        "success": True,
        "matches": standardize(pd.read_csv("./data/user.csv", index_col=0)),
        "user": standardized_user
    }


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port="8000", threads=1)