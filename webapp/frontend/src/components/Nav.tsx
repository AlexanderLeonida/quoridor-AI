import { NavLink } from "react-router-dom";

const ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/training", label: "Training", end: false },
  { to: "/selfplay", label: "Self-Play", end: false },
  { to: "/generations", label: "Generations", end: false },
  { to: "/tournament", label: "Tournament", end: false },
];

export function Nav() {
  return (
    <nav className="app-nav">
      {ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={({ isActive }) =>
            isActive ? "nav-link nav-link-active" : "nav-link"
          }
        >
          {item.label}
        </NavLink>
      ))}
    </nav>
  );
}
