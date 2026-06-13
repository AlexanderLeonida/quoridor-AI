import { useStreams } from "../useStreams";
import { SpectatorPanel } from "./SpectatorPanel";
import { TournamentPanel } from "./TournamentPanel";
import { TrainingPanel } from "./TrainingPanel";

export function Dashboard() {
  const { streams, loaded } = useStreams();

  if (!loaded) {
    return <div className="panel-empty">connecting to backend…</div>;
  }

  const hasGenerations = streams.includes("generations");
  const hasTournament = streams.includes("tournament");
  const hasMetrics = streams.includes("metrics");

  return (
    <div className="dashboard">
      <section className="dashboard-top">
        {hasMetrics ? (
          <TrainingPanel layout="strip" />
        ) : (
          <div className="panel disabled-panel">
            <div className="panel-header">
              <h2>Training Dashboard</h2>
            </div>
            <div className="panel-empty">metrics runner unavailable</div>
          </div>
        )}
      </section>

      <section className="dashboard-bottom">
        <SpectatorPanel
          streamName="selfplay"
          fallbackTitle="Self-Play"
          dense
        />
        {hasGenerations ? (
          <SpectatorPanel
            streamName="generations"
            fallbackTitle="Generations"
            dense
          />
        ) : (
          <div className="panel disabled-panel">
            <div className="panel-header">
              <h2>Generations</h2>
            </div>
            <div className="panel-empty">early checkpoint not found</div>
          </div>
        )}
        {hasTournament ? (
          <TournamentPanel dense />
        ) : (
          <div className="panel disabled-panel">
            <div className="panel-header">
              <h2>Tournament</h2>
            </div>
            <div className="panel-empty">not enough checkpoints</div>
          </div>
        )}
      </section>
    </div>
  );
}
