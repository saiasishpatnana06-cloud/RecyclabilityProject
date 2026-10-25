from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io
import os

# Safely check if database module is available
try:
    import database
    HAS_DB = True
except ImportError:
    HAS_DB = False

app = FastAPI(title="AI Waste Recyclability Intelligence API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PyTorch ImageFolder sorted classes
CLASSES = ['CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES))

MODEL_PATH = "model.pth"

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        print("SUCCESSFULLY LOADED TRAINED model.pth!")
    except Exception as e:
        print(f"Error loading model.pth: {e}")
else:
    print("WARNING: model.pth NOT FOUND IN PROJECT DIRECTORY! Operating with uninitialized weights.")

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_multilingual_guidance(material: str, contamination: str, is_mixed: bool):
    mat_key = material.upper()

    if contamination == "Heavily Contaminated":
        reason = {
            "en": f"This {mat_key.lower()} item is heavily contaminated with food or grease, making it non-recyclable.",
            "ta": f"இந்த {mat_key.lower()} பொருள் உணவு அல்லது எண்ணெயால் தீவிரமாக கறைபடிந்துள்ளது, இதனால் மறுசுழற்சி செய்ய முடியாது.",
            "hi": f"यह {mat_key.lower()} वस्तु भोजन या ग्रीस से अत्यधिक दूषित है, जिससे इसे रीसायकल नहीं किया जा सकता।",
            "te": f"ఈ {mat_key.lower()} వస్తువు ఆహారం లేదా నూనెతో తీవ్రంగా కలుషితమైంది, కాబట్టి రీసైకిల్ చేయలేము."
        }
    elif is_mixed:
        reason = {
            "en": f"This {mat_key.lower()} contains mixed non-separable materials that prevent standard recycling.",
            "ta": f"இந்த {mat_key.lower()} பொருளில் பிரிக்க முடியாத பல பொருட்கள் கலந்துள்ளதால் மறுசுழற்சி முடியாது.",
            "hi": f"इस {mat_key.lower()} वस्तु में मिश्रित गैर-पृथक्करण योग्य सामग्रियां हैं जो रीसाइक्लिंग को रोकती हैं।",
            "te": f"ఈ {mat_key.lower()} వస్తువులో వేరు చేయలేని విభిన్న పదార్థాలు ఉన్నాయి, ఇది రీసైక్లింగ్‌ను నిరోధిస్తుంది."
        }
    else:
        reason = {
            "en": f"This clean {mat_key.lower()} item is easily processed and highly suitable for recycling.",
            "ta": f"இந்த சுத்தமான {mat_key.lower()} பொருள் எளிதில் செயலாக்கப்பட்டு மறுசுழற்சிக்கு மிகவும் ஏற்றது.",
            "hi": f"यह स्वच्छ {mat_key.lower()} वस्तु आसानी से संसाधित होती है और रीसाइक्लिंग के लिए अत्यधिक उपयुक्त है।",
            "te": f"ఈ పరిశుభ్రమైన {mat_key.lower()} వస్తువు సులభంగా రీసైకిల్ చేయడానికి చాలా అనుకూలంగా ఉంటుంది."
        }

    actions = {
        "PAPER": {
            "en": "Keep dry and flat. Remove any plastic wrap or metal clips.",
            "ta": "உலர்ந்த நிலையிலும் தட்டையாகவும் வைக்கவும். பிளாஸ்டிக் அல்லது கிளிப்புகளை அகற்றவும்.",
            "hi": "सूखा और समतल रखें। प्लास्टिक रैप या मेटल क्लिप हटा दें।",
            "te": "పొడిగా మరియు ఫ్లాట్‌గా ఉంచండి. ప్లాస్టిక్ లేదా లోహపు క్లిప్‌లను తొలగించండి."
        },
        "CARDBOARD": {
            "en": "Flatten cardboard boxes completely and remove adhesive tapes.",
            "ta": "கார்ட்போர்டு பெட்டிகளை தட்டையாக மடித்து, ஒட்டும் டேப்புகளை அகற்றவும்.",
            "hi": "कार्डबोर्ड बॉक्स को पूरी तरह से चपटा करें और चिपकने वाले टेप हटा दें।",
            "te": "కార్డ్‌బోర్డ్ పెట్టెలను పూర్తిగా ఫ్లాట్‌గా మడిచి, టేప్‌లను తొలగించండి."
        },
        "METAL": {
            "en": "Rinse out residual liquids or contents from aluminum cans and tins.",
            "ta": "அலுமினிய கேன்கள் மற்றும் டின்னில் உள்ள திரவங்களை தண்ணீரில் கழுவவும்.",
            "hi": "एल्युमीनियम के डिब्बों और टिन से अवशिष्ट तरल पदार्थ धोकर साफ करें।",
            "te": "అల్యూమినియం క్యాన్లు మరియు టిన్ల నుండి ద్రవాలను కడిగి శుభ్రం చేయండి."
        },
        "PLASTIC": {
            "en": "Empty liquids, rinse clean, and crush bottles to reduce volume.",
            "ta": "திரவங்களை காலி செய்து, கழுவி, பாட்டில்களை நசுக்கி போடவும்.",
            "hi": "तरल पदार्थ खाली करें, धोकर साफ करें और बोतलों को दबाकर छोटा करें।",
            "te": "ద్రవాలను ఖాళీ చేసి, శుభ్రంగా కడిగి, సీసాలను అణిచివేయండి."
        },
        "GLASS": {
            "en": "Rinse glass jars/bottles and separate metal or plastic caps.",
            "ta": "கண்ணாடி பாட்டில்களை கழுவி, உலோக மற்றும் பிளாஸ்டிக் மூடிகளை பிரிக்கவும்.",
            "hi": "कांच के जार/बोतलों को धोएं और धातु या प्लास्टिक के ढक्कन अलग करें।",
            "te": "గాజు సీసాలను కడిగి, లోహపు లేదా ప్లాస్టిక్ మూతలను వేరు చేయండి."
        }
    }

    action = actions.get(mat_key, {
        "en": "Dispose of in designated collection bins.",
        "ta": "குறிப்பிட்ட குப்பைத்தொட்டியில் போடவும்.",
        "hi": "निर्दिष्ट कचरा डिब्बे में डालें।",
        "te": "నిర్దేశించిన చెత్త బుట్టలో వేయండి."
    })
    return {"reason": reason, "action": action}

@app.post("/predict")
async def predict_waste(
    file: UploadFile = File(...),
    contamination: str = Query("Clean"),
    is_mixed: bool = Query(False)
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor_img = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, preds = torch.max(probabilities, 0)
            detected_material = CLASSES[preds.item()]
            conf_score = round(float(conf.item()) * 100, 1)

        base_scores = {"PAPER": 95, "CARDBOARD": 90, "PLASTIC": 85, "METAL": 92, "GLASS": 88}
        score = base_scores.get(detected_material, 50)

        if contamination == "Moderately Dirty":
            score -= 20
        elif contamination == "Heavily Contaminated":
            score -= 45

        if is_mixed:
            score -= 25

        score = max(5, min(100, score))

        if score >= 75:
            status = "Highly Recyclable ✅"
        elif score >= 45:
            status = "Conditionally Recyclable (Requires Cleaning) ⚠️"
        else:
            status = "Non-Recyclable Landfill Waste ❌"

        guidance = get_multilingual_guidance(detected_material, contamination, is_mixed)

        if HAS_DB:
            try:
                database.log_prediction(
                    material=detected_material,
                    confidence=conf_score,
                    score=score,
                    status=status,
                    contamination=contamination,
                    is_mixed=is_mixed
                )
            except Exception as db_err:
                print(f"Database logging skipped: {db_err}")

        return {
            "material": detected_material,
            "confidence": conf_score,
            "score": score,
            "status": status,
            "guidance": guidance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))