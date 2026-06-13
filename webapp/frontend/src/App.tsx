import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Dashboard } from "./components/Dashboard";
import { Nav } from "./components/Nav";
import { SoloView } from "./components/SoloView";
import { SpectatorPanel } from "./components/SpectatorPanel";
import { StatsBanner } from "./components/StatsBanner";
import { TournamentPanel } from "./components/TournamentPanel";
import { TrainingPanel } from "./components/TrainingPanel";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <div className="app-title">
            <span className="brand-mark">●</span>
            <span>Quoridor RL — Live Showcase</span>
          </div>
          <div className="app-subtitle">
            A reinforcement-learning agent playing, training, and competing
            in real time
          </div>
        </header>

        <StatsBanner />

        <Nav />

        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route
              path="/training"
              element={
                <SoloView>
                  <TrainingPanel layout="grid" />
                </SoloView>
              }
            />
            <Route
              path="/selfplay"
              element={
                <SoloView>
                  <SpectatorPanel
                    streamName="selfplay"
                    fallbackTitle="Self-Play"
                  />
                </SoloView>
              }
            />
            <Route
              path="/generations"
              element={
                <SoloView>
                  <SpectatorPanel
                    streamName="generations"
                    fallbackTitle="Generations"
                  />
                </SoloView>
              }
            />
            <Route
              path="/tournament"
              element={
                <SoloView>
                  <TournamentPanel />
                </SoloView>
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <footer className="app-footer">
          <span>
            FastAPI · WebSocket streams · Quoridor self-play · MCTS
          </span>
        </footer>
      </div>
    </BrowserRouter>
  );
}
