"use strict";
/*
 * Browser-only AI for GitHub Pages: no Python server, no tunnel, no Hugging Face.
 *
 * Active only when docs/config.js sets  window.BROWSER_AI = true  and  window.API_BASE = "".
 * It answers the page's /api/* calls itself, running TensorFlow.js COCO-SSD in the browser.
 * Detects the 80 COCO classes (incl. mouse, remote, keyboard, cell phone).
 * NOT available here (needs the Python backend): ballpen, adapter/charger, connector,
 * face age/gender, expressions, gestures.
 */
(function () {
  const COCO_INFO = {"person":{"label":"Person","category":"People","description":"A person localized by the object detector. No identity is inferred."},"bicycle":{"label":"Bicycle","category":"Vehicles","description":"A human-powered two-wheeled vehicle with pedals and a frame."},"car":{"label":"Car","category":"Vehicles","description":"A road vehicle with four wheels, designed to carry a small number of people."},"motorcycle":{"label":"Motorcycle","category":"Vehicles","description":"A two-wheeled motor vehicle steered with handlebars."},"airplane":{"label":"Airplane","category":"Vehicles","description":"A powered fixed-wing aircraft."},"bus":{"label":"Bus","category":"Vehicles","description":"A large road vehicle designed to carry many passengers."},"train":{"label":"Train","category":"Vehicles","description":"A connected series of rail vehicles that runs on tracks."},"truck":{"label":"Truck","category":"Vehicles","description":"A large motor vehicle used for transporting goods."},"boat":{"label":"Boat","category":"Vehicles","description":"A watercraft designed to float and travel on water."},"traffic light":{"label":"Traffic Light","category":"Urban","description":"A signalling device that controls the flow of traffic."},"fire hydrant":{"label":"Fire Hydrant","category":"Urban","description":"A connection point through which firefighters can draw water."},"stop sign":{"label":"Stop Sign","category":"Urban","description":"A traffic sign instructing drivers to come to a complete stop."},"parking meter":{"label":"Parking Meter","category":"Urban","description":"A device used to collect payment for parking a vehicle."},"bench":{"label":"Bench","category":"Furniture","description":"A long seat designed for more than one person."},"bird":{"label":"Bird","category":"Animals","description":"A feathered, warm-blooded vertebrate, most species of which can fly."},"cat":{"label":"Cat","category":"Animals","description":"A small domesticated carnivorous mammal."},"dog":{"label":"Dog","category":"Animals","description":"A domesticated canine, kept as a companion or working animal."},"horse":{"label":"Horse","category":"Animals","description":"A large domesticated hoofed mammal used for riding and work."},"sheep":{"label":"Sheep","category":"Animals","description":"A woolly domesticated ruminant mammal."},"cow":{"label":"Cow","category":"Animals","description":"A large domesticated bovine mammal."},"elephant":{"label":"Elephant","category":"Animals","description":"A very large herbivorous mammal with a long trunk and tusks."},"bear":{"label":"Bear","category":"Animals","description":"A large heavy-bodied mammal with thick fur and a short tail."},"zebra":{"label":"Zebra","category":"Animals","description":"An African wild horse with black-and-white striped coat."},"giraffe":{"label":"Giraffe","category":"Animals","description":"A tall African mammal with a very long neck and spotted coat."},"backpack":{"label":"Backpack","category":"Accessories","description":"A bag worn on the back, usually supported by two shoulder straps."},"umbrella":{"label":"Umbrella","category":"Accessories","description":"A canopy mounted on a central pole, used for protection from rain or sun."},"handbag":{"label":"Handbag","category":"Accessories","description":"A bag used to carry personal items, typically held by hand or on the shoulder."},"tie":{"label":"Necktie","category":"Clothing","description":"A long piece of fabric worn around the neck and knotted at the collar."},"suitcase":{"label":"Suitcase","category":"Accessories","description":"A rectangular piece of luggage used to carry clothing while travelling."},"frisbee":{"label":"Frisbee","category":"Sports","description":"A circular flying disc thrown for recreation or sport."},"skis":{"label":"Skis","category":"Sports","description":"Long narrow runners worn on each foot for gliding over snow."},"snowboard":{"label":"Snowboard","category":"Sports","description":"A single board on which a rider stands sideways to descend snow."},"sports ball":{"label":"Sports Ball","category":"Sports","description":"A spherical or oval ball used in a sport or game."},"kite":{"label":"Kite","category":"Sports","description":"A tethered craft that flies in the wind."},"baseball bat":{"label":"Baseball Bat","category":"Sports","description":"A smooth wooden or metal club used to hit a baseball."},"baseball glove":{"label":"Baseball Glove","category":"Sports","description":"A padded leather glove used by baseball players to catch the ball."},"skateboard":{"label":"Skateboard","category":"Sports","description":"A board with wheels on which a person can ride or perform tricks."},"surfboard":{"label":"Surfboard","category":"Sports","description":"A narrow board used to ride waves."},"tennis racket":{"label":"Tennis Racket","category":"Sports","description":"A handled frame with a strung hitting surface, used in tennis."},"bottle":{"label":"Bottle","category":"Food & Drink","description":"A narrow-necked container used to hold liquids."},"wine glass":{"label":"Wine Glass","category":"Food & Drink","description":"A stemmed glass designed for serving wine."},"cup":{"label":"Cup","category":"Food & Drink","description":"A small open container used for drinking."},"fork":{"label":"Fork","category":"Food & Drink","description":"A utensil with prongs used for eating or serving."},"knife":{"label":"Knife","category":"Food & Drink","description":"A bladed tool or utensil used for cutting."},"spoon":{"label":"Spoon","category":"Food & Drink","description":"A utensil with a shallow bowl used for eating or stirring."},"bowl":{"label":"Bowl","category":"Food & Drink","description":"A round, deep dish used for food, liquids or mixing."},"banana":{"label":"Banana","category":"Fruits","description":"A long, curved edible fruit produced by the banana plant."},"apple":{"label":"Apple","category":"Fruits","description":"A round edible fruit produced by the apple tree."},"sandwich":{"label":"Sandwich","category":"Food","description":"Two or more slices of bread with a filling between them."},"orange":{"label":"Orange","category":"Fruits","description":"A spherical citrus fruit with a tough bright rind."},"broccoli":{"label":"Broccoli","category":"Vegetables","description":"A green vegetable with a dense head of small flower buds."},"carrot":{"label":"Carrot","category":"Vegetables","description":"A tapering orange root vegetable."},"hot dog":{"label":"Hot Dog","category":"Food","description":"A cooked sausage served in a sliced bun."},"pizza":{"label":"Pizza","category":"Food","description":"A flat round base topped with tomato, cheese and other ingredients."},"donut":{"label":"Doughnut","category":"Food","description":"A sweet fried ring of dough, often glazed or filled."},"cake":{"label":"Cake","category":"Food","description":"A sweet baked dessert made from flour, sugar and other ingredients."},"chair":{"label":"Chair","category":"Furniture","description":"A seat with a back, designed for one person."},"couch":{"label":"Couch","category":"Furniture","description":"An upholstered seat for several people."},"potted plant":{"label":"Potted Plant","category":"Home","description":"A plant growing in a container, kept indoors or outdoors."},"bed":{"label":"Bed","category":"Furniture","description":"A piece of furniture used for sleeping."},"dining table":{"label":"Dining Table","category":"Furniture","description":"A table around which people sit for meals."},"toilet":{"label":"Toilet","category":"Home","description":"A sanitary fixture used for the disposal of human waste."},"tv":{"label":"Television","category":"Electronics","description":"An electronic display used for watching television broadcasts or recordings."},"laptop":{"label":"Laptop","category":"Electronics","description":"A portable personal computer with a built-in screen and keyboard."},"mouse":{"label":"Computer Mouse","category":"Electronics","description":"A hand-held pointing device used with a computer."},"remote":{"label":"Remote Control","category":"Electronics","description":"A wireless hand-held device for controlling another device."},"keyboard":{"label":"Keyboard","category":"Electronics","description":"A panel of keys used to input text and commands into a computer."},"cell phone":{"label":"Smartphone","category":"Electronics","description":"A portable telephone with an integrated computer and touchscreen."},"microwave":{"label":"Microwave","category":"Appliances","description":"An electric appliance that heats food using microwaves."},"oven":{"label":"Oven","category":"Appliances","description":"An enclosed compartment used for baking or roasting food."},"toaster":{"label":"Toaster","category":"Appliances","description":"An electric appliance that browns bread by radiant heat."},"sink":{"label":"Sink","category":"Home","description":"A basin with a tap used for washing."},"refrigerator":{"label":"Refrigerator","category":"Appliances","description":"A cabinet or appliance that keeps food cold."},"book":{"label":"Book","category":"Stationery","description":"A set of printed or written pages bound together."},"clock":{"label":"Clock","category":"Home","description":"An instrument for measuring and displaying time."},"vase":{"label":"Vase","category":"Home","description":"An open container typically used to hold cut flowers."},"scissors":{"label":"Scissors","category":"Stationery","description":"A cutting tool with two pivoted blades."},"teddy bear":{"label":"Teddy Bear","category":"Toys","description":"A soft toy in the shape of a bear."},"hair drier":{"label":"Hair Dryer","category":"Appliances","description":"An electrical device that blows hot air to dry hair."},"toothbrush":{"label":"Toothbrush","category":"Personal Care","description":"A small brush used to clean teeth."}};
  const CATEGORY_COLORS = {"People":"#3ee9ff","Electronics":"#4d9bff","Furniture":"#a78bfa","Vehicles":"#f7b955","Animals":"#5be7a9","Fruits":"#ffb347","Vegetables":"#8bd450","Food":"#ff9366","Food & Drink":"#ff9366","Gestures":"#ffd166","Pose":"#b287ff","Expressions":"#ff7ad9","Sports":"#59d2fe","Accessories":"#f78fb3","Clothing":"#f78fb3","Home":"#74d3c6","Appliances":"#64b5f6","Stationery":"#c6a7ff","Urban":"#9db2cc","Toys":"#f9a8d4","Personal Care":"#7dd3fc","Object":"#879bb8","Unknown":"#9db2cc"};
  const TFJS_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
  const COCO_URL = "https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js";
  const FOCUS_CLASSES = new Set(["mouse", "remote", "keyboard", "cell phone"]);
  const FOCUS_CAP = 0.30; // slider can only make these MORE sensitive
  const FRUIT_CLASSES = new Set(["banana", "apple", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "sandwich"]);
  const FOOD_CATEGORIES = new Set(["Fruits", "Vegetables", "Food", "Food & Drink"]);
  const TITLES = {
    smart: ["Smart Scan", "COCO-SSD (in your browser)"],
    person: ["Person Detection", "COCO-SSD (in your browser)"],
    object: ["Object Detection", "COCO-SSD (in your browser)"],
    fruit: ["Fruit Recognition", "COCO-SSD (in your browser)"],
    gesture: ["Gesture Scanner", "needs the Python backend"],
  };
  const NOTE = "Runs fully in your browser (TensorFlow.js COCO-SSD). Detects the 80 COCO classes; pens, chargers, connectors, faces and gestures need the Python backend.";

  const bootTime = Date.now();
  let modelPromise = null;
  let modelState = "idle";
  let modelError = null;

  function colorFor(category) { return CATEGORY_COLORS[category] || "#3ee9ff"; }

  function iou(a, b) {
    const ax2 = a[0] + a[2], ay2 = a[1] + a[3], bx2 = b[0] + b[2], by2 = b[1] + b[3];
    const w = Math.max(0, Math.min(ax2, bx2) - Math.max(a[0], b[0]));
    const h = Math.max(0, Math.min(ay2, by2) - Math.max(a[1], b[1]));
    const inter = w * h;
    const union = a[2] * a[3] + b[2] * b[3] - inter;
    return union > 0 ? inter / union : 0;
  }

  function uid() { return Math.random().toString(16).slice(2, 14).padEnd(12, "0"); }

  /** Pure: COCO-SSD predictions -> the same JSON shape the Python backend returns. */
  function buildResult(predictions, dims, mode, confidence, inferenceMs) {
    const { width, height } = dims;
    const info = (name) => COCO_INFO[name] || { label: name.replace(/_/g, " "), category: "Object", description: "An object detected by COCO-SSD." };
    const notes = [];
    let kept = [];

    if (mode === "gesture") {
      notes.push("Gesture Scanner needs the Python backend (MediaPipe) and is not available in browser-only mode.");
    } else {
      for (const p of predictions) {
        const name = p.class;
        if (mode === "person" && name !== "person") continue;
        if (mode === "object" && name === "person") continue;
        if (mode === "fruit" && !FRUIT_CLASSES.has(name)) continue;
        const threshold = FOCUS_CLASSES.has(name) ? Math.min(confidence, FOCUS_CAP) : confidence;
        if (p.score < threshold) continue;
        const [x, y, w, h] = p.bbox;
        const bbox = [
          Math.max(0, x / width), Math.max(0, y / height),
          Math.min(1, w / width), Math.min(1, h / height),
        ];
        kept.push({ name, score: p.score, bbox });
      }
      // one label per object: if two classes cover the same box, the stronger wins
      kept.sort((a, b) => b.score - a.score);
      kept = kept.filter((item, i) => !kept.slice(0, i).some((o) => o.name !== item.name && iou(o.bbox, item.bbox) > 0.7));
    }

    const detections = kept.map((item) => {
      const meta = info(item.name);
      return {
        id: uid(), label: meta.label, category: meta.category,
        confidence: Math.round(item.score * 10000) / 10000,
        confidence_percent: Math.round(item.score * 1000) / 10,
        bbox: item.bbox, localized: true, description: meta.description,
        reference_key: item.name, color: colorFor(meta.category),
        source: "COCO-SSD · in-browser", attributes: {}, details: { coco_class: item.name },
        note: null, timestamp: Date.now() / 1000, reference: null,
      };
    });

    const counts = {};
    for (const d of detections) {
      const key = d.label.toLowerCase() + "|" + d.category;
      const g = counts[key] || (counts[key] = { label: d.label, category: d.category, count: 0, confidence: 0, confidence_percent: 0 });
      g.count += 1;
      if (d.confidence > g.confidence) { g.confidence = d.confidence; g.confidence_percent = d.confidence_percent; }
    }
    for (const d of detections) d.quantity = counts[d.label.toLowerCase() + "|" + d.category].count;
    const groups = Object.values(counts).sort((a, b) => b.confidence - a.confidence || a.label.localeCompare(b.label));

    const people = detections.filter((d) => d.category === "People").length;
    const fruits = detections.filter((d) => FOOD_CATEGORIES.has(d.category)).length;
    const objects = Math.max(0, detections.length - people - fruits);

    if (!detections.length && mode !== "gesture") {
      notes.push(mode === "object"
        ? "No object detected. People are excluded here: use Person Detection or Smart Scan. Move closer and add light."
        : "Nothing from the supported vocabulary was confidently detected. Try better lighting or a closer subject.");
    }
    const description = detections.length
      ? "Detected " + groups.map((g) => (g.count > 1 ? g.count + " × " : "") + g.label).join(", ") + "."
      : "The AI model did not find any supported person, object or fruit in this image. The frame may be too dark, too small, or contain items outside the model vocabulary.";

    const [title, engine] = TITLES[mode] || TITLES.smart;
    return {
      mode, detections, notes, description,
      keywords: groups.map((g) => g.label),
      meta: {
        title, engine, capability_note: NOTE, inference_ms: Math.round(inferenceMs * 10) / 10,
        image: { width, height }, server_time: Date.now() / 1000,
        people, objects, fruits, expressions: 0, groups, total_ms: Math.round(inferenceMs * 10) / 10,
        count: detections.length,
      },
      saved: 0,
    };
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const el = document.createElement("script");
      el.src = src; el.async = true;
      el.onload = resolve;
      el.onerror = () => reject(new Error("Could not load " + src + " (check your internet connection)."));
      document.head.appendChild(el);
    });
  }

  function ensureModel() {
    if (modelPromise) return modelPromise;
    modelState = "loading";
    modelPromise = (async () => {
      await loadScript(TFJS_URL);
      await loadScript(COCO_URL);
      await window.tf.ready();
      const model = await window.cocoSsd.load({ base: "mobilenet_v2" });
      modelState = "ready";
      return model;
    })().catch((error) => {
      modelState = "error";
      modelError = String(error && error.message ? error.message : error);
      modelPromise = null; // allow a retry on the next call
      throw error;
    });
    return modelPromise;
  }

  async function blobToCanvas(blob) {
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    if (bitmap.close) bitmap.close();
    return canvas;
  }

  function json(data, status = 200) {
    return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
  }

  async function handle(path, init) {
    const url = new URL(path, location.href);
    const route = url.pathname;
    const method = ((init && init.method) || "GET").toUpperCase();

    if (route === "/api/health") {
      if (modelState === "idle") ensureModel().catch(() => {});
      return json({
        ready: modelState === "ready", app: "AI Object Visualizer (browser)", version: "browser",
        models: [{ key: "yolo", name: "COCO-SSD object detector (in your browser)", state: modelState, error: modelError }],
        uptime_seconds: (Date.now() - bootTime) / 1000, device: "Browser",
        unsupported_fruits: [], supported_fruits: [...FRUIT_CLASSES],
      });
    }
    if (route === "/api/scan" && method === "POST") {
      const form = init && init.body;
      const blob = form && form.get && form.get("image");
      if (!blob) return json({ error: "No image was sent." }, 400);
      const mode = String((form.get("mode") || "smart")).toLowerCase();
      const confidence = Math.max(0.15, Math.min(0.9, parseFloat(form.get("confidence")) || 0.35));
      try {
        const model = await ensureModel();
        const canvas = await blobToCanvas(blob);
        const started = performance.now();
        const predictions = mode === "gesture" ? [] : await model.detect(canvas, 30, 0.1);
        return json(buildResult(predictions, { width: canvas.width, height: canvas.height }, mode, confidence, performance.now() - started));
      } catch (error) {
        return json({ error: "The in-browser AI could not run.", detail: String(error && error.message ? error.message : error) }, 503);
      }
    }
    if (route === "/api/stats") {
      return json({ total_scans: 0, total_detections: 0, people: 0, objects: 0, fruits: 0, expressions: 0,
        detections_24h: 0, detections_7d: 0, avg_confidence_percent: 0, daily: [], by_expression: [] });
    }
    if (route === "/api/history") return json(method === "DELETE" ? { deleted: 0 } : { items: [], total: 0 });
    if (route === "/api/reference") return json({ status: "unavailable" }, 404);
    if (route === "/api/similar") return json({ error: "Visual similarity needs the Python backend." }, 501);
    return json({ error: "Not available in browser-only mode." }, 404);
  }

  if (typeof window !== "undefined" && window.BROWSER_AI && !(window.API_BASE || "").trim()) {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init) =>
      typeof input === "string" && input.startsWith("/api/") ? handle(input, init) : nativeFetch(input, init);
    ensureModel().catch(() => {}); // start downloading the model immediately
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { buildResult, iou };
})();
