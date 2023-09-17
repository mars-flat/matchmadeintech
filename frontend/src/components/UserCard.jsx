import React from 'react';
import PieBreakdown from './PieBreakdown';
import '../styles/UserCard.css';
import downArrow from '../assets/down-arrow.svg';
import upArrow from '../assets/up-arrow.svg';
import rightArrow from '../assets/right-arrow.svg';
import leftArrow from '../assets/left-arrow.svg';

export default function UserCard(props) {
  const pfp = props.pfp;
  const name = props.name;
  const commit = props.commit;
  const showArrows = props.showArrows; // New prop to control displaying of arrows

  return (
    <div className="positioning-container">
      <div className="backdrop"></div>
      <div className={`card-container ${showArrows ? 'arrow-container' : ''}`}>
        <img src={pfp} className="img" alt="profile" />
        <div className="name">{name}</div>
        <div className="text">Commits: {commit}</div>
        <PieBreakdown />

        {showArrows && (
          <>
            {/* Arrows are displayed here */}
            <img src={upArrow} alt="up arrow" className="arrow up" />
            <img src={rightArrow} alt="right arrow" className="arrow right" />
            <img src={downArrow} alt="down arrow" className="arrow down" />
            <img src={leftArrow} alt="left arrow" className="arrow left" />
          </>
        )}
      </div>
    </div>
  );
}