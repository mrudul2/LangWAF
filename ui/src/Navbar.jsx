import { NavLink } from "react-router-dom";
import "./navbar.css";
import { useState } from "react";

const Navbar = () => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  return (
    <nav className="navbar">
      <div className="container">
        {/* Project Name */}
        <h2 className="logo">LangWAF</h2>

        {/* Navigation Links */}
        <div className="nav-elements">
          <ul>
            <li>
              <NavLink to="/">Home</NavLink>
            </li>
            {/* Model Results with Dropdown */}
            <li
              className="dropdown"
              onMouseEnter={() => setDropdownOpen(true)}
              onMouseLeave={() => setDropdownOpen(false)}
            >
              <NavLink to="/model-results">Model Results</NavLink>
              {dropdownOpen && (
                <ul className="dropdown-menu">
                  <li>
                    <NavLink to="/model1-results">Safe/Unsafe</NavLink>
                  </li>
                  <li>
                    <NavLink to="/model2-results">Language</NavLink>
                  </li>
                  <li>
                    <NavLink to="/final-model-results">Final results</NavLink>
                  </li>
                </ul>
              )}
            </li>

            <li>
              <NavLink to="/about">About</NavLink>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
