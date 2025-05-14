// =============
// Global state
// =============
let cvReady = false;
let tfReady = false;
const frameQueue = [];

// Floor detection variables
let floorBottomY = null;
let currentFloorPosition = null;

// =============
// Load OpenCV.js
// =============
let cvInitialized = false;

function initOpenCv() {
    if (cvInitialized) {
        console.log("ℹ️ OpenCV already being initialized.");
        return;
    }
    cvInitialized = true;

    const script = document.createElement("script");
    script.async = true;
    script.src = "https://docs.opencv.org/4.5.0/opencv.js "; // Fixed URL

    script.onload = () => {
        if (typeof cv !== 'undefined' && cv.onRuntimeInitialized) {
            cv.onRuntimeInitialized = () => {
                console.log("✅ OpenCV is ready!");
                cvReady = true;
            };
        } else {
            console.error("❌ Failed to initialize OpenCV: cv is undefined or runtime didn't start.");

            // Fallback: Try again after delay
            setTimeout(() => {
                console.log("🔁 Retrying OpenCV initialization...");
                cvInitialized = false; // Allow retry
                initOpenCv();
            }, 2000);
        }
    };

    script.onerror = (e) => {
        console.error("❌ Failed to load OpenCV script:", e);
        cvInitialized = false; // Allow retry
    };

    document.head.appendChild(script);
}

// =============
// Initialize MoveNet Detector
// =============
const detectorPromise = (async () => {
    console.log("[DEBUG] Initializing MoveNet detector...");
    await tf.setBackend("webgl");
    await tf.ready();
    const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
    });
    console.log("[DEBUG] Detector ready.");
    tfReady = true;
    return detector;
})();

// =============
// Canvas for both detectors
// =============
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

// =============
// Queue frames until both CV & TF are ready
// =============
function queueFrame(base64) {
    frameQueue.push(base64);
    processQueuedFrames();
}

function processQueuedFrames() {
    while (frameQueue.length > 0 && cvReady && tfReady) {
        const base64 = frameQueue.shift();
        processBothDetectors(base64);
    }
}

// =============
// Unified Frame Processor
// =============
async function processBothDetectors(base64) {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = "data:image/jpeg;base64," + base64;

    image.onload = async () => {
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);

        // =============
        // Run OpenCV Floor Detection
        // =============
        try {
            let src = cv.imread(canvas);
            let gray = new cv.Mat();
            let edges = new cv.Mat();

            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            cv.Canny(gray, edges, 50, 150);

            let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
            cv.dilate(edges, edges, kernel, new cv.Point(-1, -1), 5);
            cv.erode(edges, edges, kernel, new cv.Point(-1, -1), 3);

            let lines = new cv.Mat();
            cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 20, 20, 10);

            let leftLines = [];
            let rightLines = [];

            for (let i = 0; i < lines.rows; ++i) {
                let x1 = lines.data32S[i * 4];
                let y1 = lines.data32S[i * 4 + 1];
                let x2 = lines.data32S[i * 4 + 2];
                let y2 = lines.data32S[i * 4 + 3];

                let dx = x2 - x1;
                let dy = y2 - y1;
                let length = Math.sqrt(dx * dx + dy * dy);
                if (length < 20) continue;

                let slope = (dy / (dx || 0.0001)).toFixed(2);
                if (Math.abs(slope) < 0.1 || Math.abs(parseFloat(slope)) > 10) continue;

                if (parseFloat(slope) > 0) {
                    leftLines.push({ x1, y1, x2, y2 });
                } else {
                    rightLines.push({ x1, y1, x2, y2 });
                }
            }

            let bestLeft = getBestLine(leftLines);
            let bestRight = getBestLine(rightLines);

            if (bestLeft && bestRight) {
                floorBottomY = (bestLeft.y2 + bestRight.y1) / 2;

                let normalizedY = (floorBottomY / canvas.height) * 2 - 1;
                let yUnity = -normalizedY * 0.5;

                // const cameraEl = document.getElementById('camera');
                // if (!cameraEl) return;

                // const camDir = new THREE.Vector3(0, 0, -1);
                // camDir.applyQuaternion(cameraEl.object3D.quaternion);

                // currentFloorPosition = {
                //     x: camDir.x * 2,
                //     y: yUnity,
                //     z: camDir.z * 2
                // };
                // console.error("Detecting Floor:", JSON.stringify(currentFloorPosition));
                // // Send floor position to Unity
                // if (window.UnityInstance) {
                //     UnityInstance.SendMessage("FloorDetector", "OnReceiveFloorPosition", JSON.stringify(currentFloorPosition));
                // }
            }

            // Cleanup
            if (src) src.delete();
            if (gray) gray.delete();
            if (edges) edges.delete();
            if (lines) lines.delete();
        } catch (err) {
            console.error("OpenCV Error:", err);
        }

        // =============
        // Run MoveNet Foot Detection
        // =============
        try {
            const detector = await detectorPromise;
            const poses = await detector.estimatePoses(canvas);
            if (poses.length === 0) {
                console.warn("[DEBUG] No poses detected.");
                return;
            }

            const keypoints = poses[0].keypoints;
            const leftAnkle = keypoints[15];  // left ankle index
            const rightAnkle = keypoints[16]; // right ankle index

            const foot = (leftAnkle?.score ?? 0) > (rightAnkle?.score ?? 0) ? leftAnkle : rightAnkle;

            if (foot && foot.score > 0.3) {
                const normalized = {
                    x: foot.x / canvas.width,
                    y: foot.y / canvas.height
                };

                console.error("Detecting foot:", JSON.stringify(normalized));
                if (window.UnityInstance) {
                    window.unityInstance.SendMessage("FootCube", "OnReceiveFootPosition", JSON.stringify(normalized));
                }
            }
        } catch (err) {
            console.error("MoveNet Error:", err);
        }
    };
}

// Utility: Get longest line
function getBestLine(lines) {
    if (!lines || lines.length === 0) return null;
    return lines.reduce((a, b) => {
        let lenA = Math.hypot(a.x2 - a.x1, a.y2 - a.y1);
        let lenB = Math.hypot(b.x2 - b.x1, b.y2 - b.y1);
        return lenA > lenB ? a : b;
    });
}

// =============
// Expose to Unity
// =============
window.ReceiveWebcamFrame = window.ReceiveWebcamFrameFloor = function (base64) {
    processBothDetectors(base64);
};

// =============
// Kickstart everything
// =============
initOpenCv();

// =============
// Start A-Frame Scene
// =============
window.cameraReady = function () {
    startAFrameScene();
};

function startAFrameScene() {
    console.log("📷 Initializing A-Frame AR scene...");

    const arScene = document.createElement('a-scene');
    arScene.setAttribute("embedded", "");
    arScene.setAttribute("arjs", "sourceType: webcam; debugUIEnabled: false");
    arScene.setAttribute("vr-mode-ui", "enabled: false");
    arScene.style.zIndex = "1";
    arScene.style.position = "absolute";
    arScene.style.top = "0";
    arScene.style.left = "0";
    arScene.style.width = "0vw";
    arScene.style.height = "0vh";

    // Add camera
    const cameraEntity = document.createElement('a-entity');
    cameraEntity.setAttribute('id', 'camera');
    cameraEntity.setAttribute('camera', '');
    cameraEntity.setAttribute('look-controls', '');
    cameraEntity.setAttribute('cameratransform', ''); // Custom component

    arScene.appendChild(cameraEntity);
    document.body.appendChild(arScene);
}

// Component to send camera data to Unity every frame
AFRAME.registerComponent('cameratransform', {
    schema: {},

    tock: function () {
        const el = this.el;
        const camera = el.components.camera.camera;

        if (!camera) return;

        // Get camera position and rotation
        let position = new THREE.Vector3();
        let quaternion = new THREE.Quaternion();
        let scale = new THREE.Vector3();

        el.object3D.matrix.clone().decompose(position, quaternion, scale);

        // Convert to array string
        const posStr = position.toArray().join(",");
        const rotStr = quaternion.toArray().join(",");
        const projStr = [...camera.projectionMatrix.elements].join(",");

        // Send to Unity
        if (window.UnityInstance) {
            UnityInstance.SendMessage("MainCamera", "SetPosition", posStr);
            UnityInstance.SendMessage("MainCamera", "SetRotation", rotStr);
            UnityInstance.SendMessage("MainCamera", "SetProjection", projStr);

            // Optional: Send canvas size
            const canvases = document.getElementsByTagName('canvas');
            if (canvases.length > 0) {
                const w = canvases[0].width;
                const h = canvases[0].height;
                // UnityInstance.SendMessage("Canvas", "SetSize", `${w},${h}`);
            }
        }
    }
});
