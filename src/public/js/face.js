const faceWrapper = document.querySelector(".faceWrapper");
const loadWrapper = document.querySelector(".loadWrapper");
const video = document.getElementById("faceRecognition");
const faceButton = document.getElementById("faceButton");
const main = document.getElementById("welcome");

faceWrapper.hidden = true;
faceButton.hidden = true;
main.hidden = true;

// nickname 초기화
if (localStorage.getItem("nickname")) {
  localStorage.removeItem("nickname");
}

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("public/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("public/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("public/models"), //heavier/accurate version of tiny face detector
]).then(start);

function start() {
  navigator.getUserMedia(
    { video: {} },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );

  console.log("video added");
  recognizeFaces();
}

async function recognizeFaces() {
  const labeledDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.7);
  let count = 0;

  video.addEventListener("play", async () => {
    console.log("Playing");

    faceWrapper.hidden = false;
    loadWrapper.hidden = true;

    const canvas = faceapi.createCanvasFromMedia(video);
    faceWrapper.append(canvas);

    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
      const detections = await faceapi
        .detectAllFaces(video)
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

      const results = resizedDetections.map((d) => {
        return faceMatcher.findBestMatch(d.descriptor);
      });
      // 얼굴인식이 성공하면 LocalStorage에 저장
      if (results[0]?._label) {
        count++;
      }

      if (results[0]?._label !== "unknown" && count === 1) {
        localStorage.setItem("nickname", results[0]._label);
      }
      const nickname = localStorage.getItem("nickname");
      if (nickname !== null && nickname !== "unknown" && nickname !== "") {
        faceButton.hidden = false;
        faceButton.addEventListener("click", () => {
          faceWrapper.hidden = true;
          main.hidden = false;
        });
      }
      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: result.toString(),
        });
        drawBox.draw(canvas);
      });
    }, 100);
  });

  // Video 자동재생처럼 보이기 위한 연출
  await video.play();
}

function loadLabeledImages() {
  const labels = ["Minchan Lee", "Geunhyuk Yang"]; // for WebCam
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `public/labeled_images/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        // console.log(label + i + JSOsN.stringify(detections));
        descriptions.push(detections.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
