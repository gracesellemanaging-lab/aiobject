"""Human-readable knowledge, categories and visual-reference lookup metadata.

The detection models return technical class names.  This module turns those
names into labels a human can read, groups them into categories and attaches a
short, accurate description.  It also curates Wikimedia lookup queries so the
reference image shown by the UI matches the detected item.
"""
from __future__ import annotations

from typing import Any


def _info(label: str, category: str, description: str, reference: str | None = None,
          query: str | None = None) -> dict[str, Any]:
    return {
        "label": label,
        "category": category,
        "description": description,
        "reference": reference or label,
        "query": query or label,
    }


COCO_INFO: dict[str, dict[str, Any]] = {
    "person": _info("Person", "People", "A person localized by the object detector. No identity is inferred.", "Human", "person portrait"),
    "bicycle": _info("Bicycle", "Vehicles", "A human-powered two-wheeled vehicle with pedals and a frame.", "Bicycle"),
    "car": _info("Car", "Vehicles", "A road vehicle with four wheels, designed to carry a small number of people.", "Car", "automobile"),
    "motorcycle": _info("Motorcycle", "Vehicles", "A two-wheeled motor vehicle steered with handlebars.", "Motorcycle"),
    "airplane": _info("Airplane", "Vehicles", "A powered fixed-wing aircraft.", "Airplane"),
    "bus": _info("Bus", "Vehicles", "A large road vehicle designed to carry many passengers.", "Bus"),
    "train": _info("Train", "Vehicles", "A connected series of rail vehicles that runs on tracks.", "Train"),
    "truck": _info("Truck", "Vehicles", "A large motor vehicle used for transporting goods.", "Truck"),
    "boat": _info("Boat", "Vehicles", "A watercraft designed to float and travel on water.", "Boat"),
    "traffic light": _info("Traffic Light", "Urban", "A signalling device that controls the flow of traffic.", "Traffic light"),
    "fire hydrant": _info("Fire Hydrant", "Urban", "A connection point through which firefighters can draw water.", "Fire hydrant"),
    "stop sign": _info("Stop Sign", "Urban", "A traffic sign instructing drivers to come to a complete stop.", "Stop sign"),
    "parking meter": _info("Parking Meter", "Urban", "A device used to collect payment for parking a vehicle.", "Parking meter"),
    "bench": _info("Bench", "Furniture", "A long seat designed for more than one person.", "Bench (furniture)"),
    "bird": _info("Bird", "Animals", "A feathered, warm-blooded vertebrate, most species of which can fly.", "Bird"),
    "cat": _info("Cat", "Animals", "A small domesticated carnivorous mammal.", "Cat"),
    "dog": _info("Dog", "Animals", "A domesticated canine, kept as a companion or working animal.", "Dog"),
    "horse": _info("Horse", "Animals", "A large domesticated hoofed mammal used for riding and work.", "Horse"),
    "sheep": _info("Sheep", "Animals", "A woolly domesticated ruminant mammal.", "Sheep"),
    "cow": _info("Cow", "Animals", "A large domesticated bovine mammal.", "Cattle", "cow"),
    "elephant": _info("Elephant", "Animals", "A very large herbivorous mammal with a long trunk and tusks.", "Elephant"),
    "bear": _info("Bear", "Animals", "A large heavy-bodied mammal with thick fur and a short tail.", "Bear"),
    "zebra": _info("Zebra", "Animals", "An African wild horse with black-and-white striped coat.", "Zebra"),
    "giraffe": _info("Giraffe", "Animals", "A tall African mammal with a very long neck and spotted coat.", "Giraffe"),
    "backpack": _info("Backpack", "Accessories", "A bag worn on the back, usually supported by two shoulder straps.", "Backpack"),
    "umbrella": _info("Umbrella", "Accessories", "A canopy mounted on a central pole, used for protection from rain or sun.", "Umbrella"),
    "handbag": _info("Handbag", "Accessories", "A bag used to carry personal items, typically held by hand or on the shoulder.", "Handbag"),
    "tie": _info("Necktie", "Clothing", "A long piece of fabric worn around the neck and knotted at the collar.", "Necktie"),
    "suitcase": _info("Suitcase", "Accessories", "A rectangular piece of luggage used to carry clothing while travelling.", "Suitcase"),
    "frisbee": _info("Frisbee", "Sports", "A circular flying disc thrown for recreation or sport.", "Frisbee"),
    "skis": _info("Skis", "Sports", "Long narrow runners worn on each foot for gliding over snow.", "Ski"),
    "snowboard": _info("Snowboard", "Sports", "A single board on which a rider stands sideways to descend snow.", "Snowboard"),
    "sports ball": _info("Sports Ball", "Sports", "A spherical or oval ball used in a sport or game.", "Ball"),
    "kite": _info("Kite", "Sports", "A tethered craft that flies in the wind.", "Kite"),
    "baseball bat": _info("Baseball Bat", "Sports", "A smooth wooden or metal club used to hit a baseball.", "Baseball bat"),
    "baseball glove": _info("Baseball Glove", "Sports", "A padded leather glove used by baseball players to catch the ball.", "Baseball glove"),
    "skateboard": _info("Skateboard", "Sports", "A board with wheels on which a person can ride or perform tricks.", "Skateboard"),
    "surfboard": _info("Surfboard", "Sports", "A narrow board used to ride waves.", "Surfboard"),
    "tennis racket": _info("Tennis Racket", "Sports", "A handled frame with a strung hitting surface, used in tennis.", "Racket (sports equipment)"),
    "bottle": _info("Bottle", "Food & Drink", "A narrow-necked container used to hold liquids.", "Bottle"),
    "wine glass": _info("Wine Glass", "Food & Drink", "A stemmed glass designed for serving wine.", "Wine glass"),
    "cup": _info("Cup", "Food & Drink", "A small open container used for drinking.", "Cup"),
    "fork": _info("Fork", "Food & Drink", "A utensil with prongs used for eating or serving.", "Fork"),
    "knife": _info("Knife", "Food & Drink", "A bladed tool or utensil used for cutting.", "Knife"),
    "spoon": _info("Spoon", "Food & Drink", "A utensil with a shallow bowl used for eating or stirring.", "Spoon"),
    "bowl": _info("Bowl", "Food & Drink", "A round, deep dish used for food, liquids or mixing.", "Bowl"),
    "banana": _info("Banana", "Fruits", "A long, curved edible fruit produced by the banana plant.", "Banana", "banana fruit"),
    "apple": _info("Apple", "Fruits", "A round edible fruit produced by the apple tree.", "Apple", "apple fruit"),
    "sandwich": _info("Sandwich", "Food", "Two or more slices of bread with a filling between them.", "Sandwich"),
    "orange": _info("Orange", "Fruits", "A spherical citrus fruit with a tough bright rind.", "Orange (fruit)", "orange fruit"),
    "broccoli": _info("Broccoli", "Vegetables", "A green vegetable with a dense head of small flower buds.", "Broccoli"),
    "carrot": _info("Carrot", "Vegetables", "A tapering orange root vegetable.", "Carrot"),
    "hot dog": _info("Hot Dog", "Food", "A cooked sausage served in a sliced bun.", "Hot dog"),
    "pizza": _info("Pizza", "Food", "A flat round base topped with tomato, cheese and other ingredients.", "Pizza"),
    "donut": _info("Doughnut", "Food", "A sweet fried ring of dough, often glazed or filled.", "Doughnut", "donut"),
    "cake": _info("Cake", "Food", "A sweet baked dessert made from flour, sugar and other ingredients.", "Cake"),
    "chair": _info("Chair", "Furniture", "A seat with a back, designed for one person.", "Chair"),
    "couch": _info("Couch", "Furniture", "An upholstered seat for several people.", "Couch", "sofa"),
    "potted plant": _info("Potted Plant", "Home", "A plant growing in a container, kept indoors or outdoors.", "Houseplant"),
    "bed": _info("Bed", "Furniture", "A piece of furniture used for sleeping.", "Bed"),
    "dining table": _info("Dining Table", "Furniture", "A table around which people sit for meals.", "Table (furniture)", "dining table"),
    "toilet": _info("Toilet", "Home", "A sanitary fixture used for the disposal of human waste.", "Toilet"),
    "tv": _info("Television", "Electronics", "An electronic display used for watching television broadcasts or recordings.", "Television", "television screen"),
    "laptop": _info("Laptop", "Electronics", "A portable personal computer with a built-in screen and keyboard.", "Laptop"),
    "mouse": _info("Computer Mouse", "Electronics", "A hand-held pointing device used with a computer.", "Computer mouse", "computer mouse"),
    "remote": _info("Remote Control", "Electronics", "A wireless hand-held device for controlling another device.", "Remote control"),
    "keyboard": _info("Keyboard", "Electronics", "A panel of keys used to input text and commands into a computer.", "Computer keyboard", "computer keyboard"),
    "cell phone": _info("Smartphone", "Electronics", "A portable telephone with an integrated computer and touchscreen.", "Smartphone"),
    "microwave": _info("Microwave", "Appliances", "An electric appliance that heats food using microwaves.", "Microwave oven"),
    "oven": _info("Oven", "Appliances", "An enclosed compartment used for baking or roasting food.", "Oven"),
    "toaster": _info("Toaster", "Appliances", "An electric appliance that browns bread by radiant heat.", "Toaster"),
    "sink": _info("Sink", "Home", "A basin with a tap used for washing.", "Sink"),
    "refrigerator": _info("Refrigerator", "Appliances", "A cabinet or appliance that keeps food cold.", "Refrigerator", "refrigerator appliance"),
    "book": _info("Book", "Stationery", "A set of printed or written pages bound together.", "Book"),
    "clock": _info("Clock", "Home", "An instrument for measuring and displaying time.", "Clock"),
    "vase": _info("Vase", "Home", "An open container typically used to hold cut flowers.", "Vase"),
    "scissors": _info("Scissors", "Stationery", "A cutting tool with two pivoted blades.", "Scissors"),
    "teddy bear": _info("Teddy Bear", "Toys", "A soft toy in the shape of a bear.", "Teddy bear"),
    "hair drier": _info("Hair Dryer", "Appliances", "An electrical device that blows hot air to dry hair.", "Hair dryer", "hair dryer"),
    "toothbrush": _info("Toothbrush", "Personal Care", "A small brush used to clean teeth.", "Toothbrush"),
}

# ImageNet class names mapped to a fruit / vegetable / food display item.
IMAGENET_FOOD_INFO: dict[str, dict[str, Any]] = {
    "granny smith": _info("Apple", "Fruits", "A round edible fruit produced by the apple tree; this class is the green Granny Smith variety.", "Apple", "green apple fruit"),
    "strawberry": _info("Strawberry", "Fruits", "A small red fruit with tiny seeds on its surface and a sweet-tart flavor.", "Strawberry", "strawberry fruit"),
    "orange": _info("Orange", "Fruits", "A spherical citrus fruit with a tough bright rind and juicy pulp.", "Orange (fruit)", "orange fruit"),
    "lemon": _info("Lemon", "Fruits", "A yellow citrus fruit with a sour, aromatic juice.", "Lemon", "lemon fruit"),
    "fig": _info("Fig", "Fruits", "A soft, sweet fruit with a chewy skin and many tiny seeds inside.", "Common fig", "fig fruit"),
    "pineapple": _info("Pineapple", "Fruits", "A tropical fruit with a spiky crown and sweet yellow flesh.", "Pineapple", "pineapple fruit"),
    "banana": _info("Banana", "Fruits", "A long curved tropical fruit with soft, sweet flesh.", "Banana", "banana fruit"),
    "jackfruit": _info("Jackfruit", "Fruits", "A very large tropical fruit with sweet yellow edible bulbs.", "Jackfruit", "jackfruit fruit"),
    "custard apple": _info("Custard Apple", "Fruits", "A heart-shaped tropical fruit with creamy, sweet flesh.", "Custard apple", "custard apple fruit"),
    "pomegranate": _info("Pomegranate", "Fruits", "A round fruit filled with juicy red seeds called arils.", "Pomegranate", "pomegranate fruit"),
    "avocado": _info("Avocado", "Fruits", "A creamy green tropical fruit with a large central stone and thick skin.", "Avocado", "avocado fruit"),
    "cucumber": _info("Cucumber", "Vegetables", "A long green vegetable with a mild, watery flesh.", "Cucumber"),
    "bell pepper": _info("Bell Pepper", "Vegetables", "A glossy hollow vegetable that comes in green, red, yellow and orange varieties.", "Bell pepper"),
    "broccoli": _info("Broccoli", "Vegetables", "A green vegetable with a dense head of small edible flower buds.", "Broccoli"),
    "cauliflower": _info("Cauliflower", "Vegetables", "A vegetable with a white compact head of undeveloped flower buds.", "Cauliflower"),
    "zucchini": _info("Zucchini", "Vegetables", "A summer squash with mild flesh and edible skin.", "Zucchini", "zucchini courgette"),
    "spaghetti squash": _info("Spaghetti Squash", "Vegetables", "A winter squash whose cooked flesh separates into spaghetti-like strands.", "Spaghetti squash"),
    "butternut squash": _info("Butternut Squash", "Vegetables", "A sweet, orange-fleshed winter squash with a pale tan rind.", "Butternut squash"),
    "acorn squash": _info("Acorn Squash", "Vegetables", "A small acorn-shaped winter squash with dark green ribbed skin.", "Acorn squash"),
    "artichoke": _info("Artichoke", "Vegetables", "An edible thistle with layered leaves and a tender heart.", "Artichoke"),
    "cardoon": _info("Cardoon", "Vegetables", "A thistle-like vegetable related to the artichoke.", "Cardoon"),
    "mushroom": _info("Mushroom", "Food", "The fleshy fruiting body of a fungus, used in cooking.", "Mushroom", "edible mushroom"),
    "head cabbage": _info("Cabbage", "Vegetables", "A leafy green or purple vegetable that forms a compact head.", "Cabbage", "cabbage vegetable"),
    "ear of corn": _info("Corn", "Vegetables", "A cereal grain with kernels arranged on a central cob.", "Maize", "corn cob"),
    "cheeseburger": _info("Cheeseburger", "Food", "A hamburger topped with melted cheese.", "Cheeseburger"),
    "hotdog": _info("Hot Dog", "Food", "A cooked sausage served in a sliced bun.", "Hot dog"),
    "pizza": _info("Pizza", "Food", "A flat round base topped with tomato, cheese and other ingredients.", "Pizza"),
    "burrito": _info("Burrito", "Food", "A rolled tortilla filled with meat, beans, cheese or other ingredients.", "Burrito"),
    "pretzel": _info("Pretzel", "Food", "A twisted baked bread product, often salted.", "Pretzel"),
    "bagel": _info("Bagel", "Food", "A ring-shaped bread roll that is boiled before baking.", "Bagel"),
    "french loaf": _info("Bread", "Food", "A baked food made from a dough of flour and water.", "Bread"),
    "ice cream": _info("Ice Cream", "Food", "A sweet frozen dessert made from dairy or plant-based ingredients.", "Ice cream"),
    "ice lolly": _info("Ice Pop", "Food", "A frozen snack made from flavoured water or juice on a stick.", "Ice pop", "popsicle"),
    "guacamole": _info("Guacamole", "Food", "A Mexican dip made from mashed avocado, lime and seasonings.", "Guacamole"),
    "mashed potato": _info("Mashed Potato", "Food", "Cooked potatoes mashed with butter or milk until smooth.", "Mashed potato"),
    "carbonara": _info("Carbonara", "Food", "An Italian pasta dish with eggs, cheese and cured pork.", "Carbonara"),
    "espresso": _info("Espresso", "Food & Drink", "A concentrated coffee beverage brewed by forcing hot water through finely ground beans.", "Espresso"),
    "red wine": _info("Red Wine", "Food & Drink", "An alcoholic drink made from fermented dark grapes.", "Red wine"),
    "eggnog": _info("Eggnog", "Food & Drink", "A sweetened dairy-based drink traditionally made with eggs and milk.", "Eggnog"),
    "trifle": _info("Trifle", "Food", "A layered English dessert with sponge, fruit, custard and cream.", "Trifle (dessert)", "trifle dessert"),
    "hot pot": _info("Hot Pot", "Food", "A communal meal in which diners cook ingredients in a simmering pot of broth.", "Hot pot"),
    "meat loaf": _info("Meatloaf", "Food", "A dish of ground meat formed into a loaf shape and baked.", "Meatloaf"),
    "dough": _info("Dough", "Food", "A thick mixture of flour and liquid used to make bread or pastry.", "Dough"),
}

# Classes the lightweight fruit scanner can localize directly with YOLO.
YOLO_FRUIT_CLASSES = {"banana", "apple", "orange", "broccoli", "carrot"}

FRUIT_SCAN_SUPPORTED = [
    "Apple", "Banana", "Orange", "Lemon", "Pineapple", "Strawberry",
    "Pomegranate", "Fig", "Jackfruit", "Custard Apple", "Avocado",
    "Cucumber", "Bell Pepper", "Broccoli", "Cauliflower", "Zucchini",
    "Mushroom", "Artichoke", "Corn", "Cabbage",
]


GESTURE_INFO: dict[str, dict[str, Any]] = {
    "smile": _info(
        "Happy / Smiling", "Expressions",
        "Visible smile-related facial features were detected. This is a visual expression estimate, not a claim about the person's emotions or personality.",
        "Smile", "person smiling",
    ),
    "neutral": _info(
        "Neutral Expression", "Expressions",
        "The visible facial features are relaxed. A neutral expression says nothing definitive about how someone feels.",
        "Facial expression", "neutral facial expression",
    ),
    "sad": _info(
        "Sad Expression", "Expressions",
        "A downturned mouth and lowered brows were visible. This is a facial-configuration estimate, not a diagnosis or a claim about the person's feelings.",
        "Sadness", "sad facial expression",
    ),
    "surprised": _info(
        "Surprised Expression", "Expressions",
        "An open mouth and raised inner brows were detected. This is a visible facial configuration, not a definitive emotion reading.",
        "Surprise (emotion)", "surprised face expression",
    ),
    "waving": _info(
        "Waving", "Gestures",
        "A raised hand with horizontal back-and-forth motion was observed across multiple frames.",
        "Hand wave", "person waving hand",
    ),
    "thumbs-up": _info(
        "Thumbs Up", "Gestures",
        "A hand with the thumb extended upward and the other fingers folded was observed.",
        "Thumb signal", "thumbs up gesture",
    ),
    "thumbs-down": _info(
        "Thumbs Down", "Gestures",
        "A hand with the thumb extended downward and the other fingers folded was observed.",
        "Thumb signal", "thumbs down gesture",
    ),
    "raising-one-hand": _info(
        "Raising One Hand", "Gestures",
        "One wrist was detected above shoulder level.",
        "Hands up", "raising hand gesture",
    ),
    "raising-both-hands": _info(
        "Raising Both Hands", "Gestures",
        "Both wrists were detected above shoulder level.",
        "Hands up", "raising both hands",
    ),
    "pointing": _info(
        "Pointing", "Gestures",
        "A hand was detected with the index finger extended and the other fingers folded.",
        "Pointing", "pointing finger gesture",
    ),
    "clapping": _info(
        "Clapping", "Gestures",
        "Both hands were observed meeting and separating repeatedly near chest height across multiple frames.",
        "Applause", "clapping hands",
    ),
    "standing": _info(
        "Standing", "Pose",
        "A mostly upright body pose with visible, relatively straight legs.",
        "Standing", "person standing",
    ),
    "sitting": _info(
        "Sitting", "Pose",
        "A body pose consistent with bent hips and knees.",
        "Sitting", "person sitting",
    ),
    "walking": _info(
        "Walking", "Pose",
        "Alternating leg movement was observed across multiple live frames, consistent with walking.",
        "Walking", "person walking",
    ),
}

# Values used to make the UI clearly communicate model limits.
CAPABILITY_NOTES = {
    "person": "Man/woman and age ranges are uncertain AI estimates computed from visible facial features. They are not facts, exact ages or identity checks.",
    "object": "Universal scanner uses YOLOv8n trained on the 80 COCO object classes. Objects outside this vocabulary are not claimed as recognized.",
    "fruit": "Fruit scanner combines YOLO localization with an ImageNet food classifier. Fruits outside its vocabulary (for example mango, grapes, watermelon, papaya or dragon fruit) are marked as unsupported instead of guessed.",
    "gesture": "Gestures and expressions are estimated from body, hand and facial landmarks. Temporal gestures need several live frames. Expressions do not reveal true emotions or personality.",
    "smart": "Smart Scan combines person, object, expression and age-range detection. Visual estimates are approximate and are labeled accordingly.",
}

SCANNER_META = {
    "smart": {"title": "Smart Scan", "engine": "YOLOv8n + FairFace (age + gender) + MediaPipe Face", "note": CAPABILITY_NOTES["smart"]},
    "person": {"title": "Person Detection", "engine": "YOLOv8n + FairFace (age + gender)", "note": CAPABILITY_NOTES["person"]},
    "object": {"title": "Object Detection", "engine": "YOLOv8n · 80 COCO classes", "note": CAPABILITY_NOTES["object"]},
    "fruit": {"title": "Fruit Recognition", "engine": "YOLOv8n + MobileNetV3", "note": CAPABILITY_NOTES["fruit"]},
    "gesture": {"title": "Gesture Scanner", "engine": "MediaPipe Pose · Hands · Face", "note": CAPABILITY_NOTES["gesture"]},
}

REFERENCE_INFO: dict[str, dict[str, str]] = {
    "person": {"title": "Human", "query": "person portrait"},
    "human": {"title": "Human", "query": "person portrait"},
    "smile": {"title": "Smile", "query": "person smiling"},
    "neutral": {"title": "Facial expression", "query": "neutral facial expression"},
    "sad": {"title": "Sadness", "query": "sad facial expression"},
    "surprised": {"title": "Surprise (emotion)", "query": "surprised facial expression"},
    "waving": {"title": "Hand wave", "query": "person waving hand"},
    "thumbs-up": {"title": "Thumb signal", "query": "thumbs up gesture"},
    "thumbs-down": {"title": "Thumb signal", "query": "thumbs down gesture"},
    "raising-one-hand": {"title": "Hands up", "query": "raising hand gesture"},
    "raising-both-hands": {"title": "Hands up", "query": "raising both hands"},
    "pointing": {"title": "Pointing", "query": "pointing finger gesture"},
    "clapping": {"title": "Applause", "query": "clapping hands"},
    "standing": {"title": "Standing", "query": "person standing"},
    "sitting": {"title": "Sitting", "query": "person sitting"},
    "walking": {"title": "Walking", "query": "person walking"},
}


def reference_info(key: str, fallback_label: str = "", fallback_query: str = "") -> dict[str, str]:
    """Return curated reference metadata or a sensible fallback."""
    if key in REFERENCE_INFO:
        return REFERENCE_INFO[key]
    if key in COCO_INFO:
        info = COCO_INFO[key]
        return {"title": info["reference"], "query": info["query"]}
    if key in IMAGENET_FOOD_INFO:
        info = IMAGENET_FOOD_INFO[key]
        return {"title": info["reference"], "query": info["query"]}
    if key in GESTURE_INFO:
        info = GESTURE_INFO[key]
        return {"title": info["reference"], "query": info["query"]}
    label = fallback_label or key.replace("-", " ").replace("_", " ").strip().title()
    return {"title": label, "query": fallback_query or label}


def lookup(raw_label: str) -> dict[str, Any] | None:
    """Look up display metadata for a COCO, food or gesture class key."""
    key = raw_label.lower().strip()
    for source in (COCO_INFO, IMAGENET_FOOD_INFO, GESTURE_INFO):
        if key in source:
            return source[key]
    return None
