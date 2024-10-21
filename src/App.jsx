import { useState, useCallback } from "react";
import "./App.css";
import { useModel } from "./hooks/useModel";

function App() {
    const [webgpuEnabled, setWebgpuEnabled] = useState(false);
    const [profileEnabled, setProfileEnabled] = useState(false);

    const { run } = useModel();

    const onRun = useCallback(async () => {

        await run(true, profileEnabled); // wgpu
        await run(false, false); // cpu, profiling disabled
        
    }, [run, webgpuEnabled, profileEnabled]);

    return (
        <main style={{ display: "flex", flexDirection: "column" }}>
            <button onClick={onRun}>Run Demucs (first wasm then gpu)</button>
            <div>
                <input
                    id="webgpu"
                    type="checkbox"
                    disabled
                    onChange={(e) => setWebgpuEnabled(e.target.checked)}
                />
                <label htmlFor="webgpu">Enable WebGPU</label>
            </div>
            <div>
                <input
                    id="profile"
                    type="checkbox"
                    onChange={(e) => setProfileEnabled(e.target.checked)}
                />
                <label htmlFor="profile">Enable profiling</label>
            </div>
        </main>
    );
}

export default App;
