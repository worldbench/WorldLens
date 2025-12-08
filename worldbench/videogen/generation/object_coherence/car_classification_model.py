from transformers import ViTImageProcessor, ViTForImageClassification
import re
from PIL import Image
import torch
import easydict

CAR_PATTERNS = [
    r"\bcar\b", r"\btaxi\b", r"\bcab\b", r"\bminivan\b", r"\bvan\b", r"\bjeep\b",
    r"\blimousine\b", r"\bpickup\b", r"\brace car\b", r"\bpolice van\b",
    r"\bambulance\b", r"\bfire (engine|truck)\b", r"\brv\b", r"\bstation wagon\b",
    r"\bconvertible\b", r"\bsports car\b", r"\bbeach wagon\b", r"\bmodel t\b"
]
PERSON_PATTERNS = [
    r"\bperson\b", r"\bman\b", r"\bwoman\b", r"\bboy\b", r"\bgirl\b",
    r"\bbride\b", r"\bgroom\b", r"\bscuba diver\b", r"\bdiver\b",
    r"\bswimmer\b", r"\bskier\b", r"\bsnowboarder\b", r"\bbaseball player\b",
    r"\bbasketball player\b", r"\bfootball player\b", r"\bmilitary uniform\b",
    r"\bpolice officer\b", r"\bfirefighter\b", r"\bmonk\b", r"\bnun\b", r"\bpriest\b"
]

def compile_patterns(patterns):
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]

re_car = compile_patterns(CAR_PATTERNS)
re_person = compile_patterns(PERSON_PATTERNS)

def match_group(label, regex_list):
    return any(r.search(label) for r in regex_list)

def build_index_sets(id2label):
    car_ids, person_ids = [], []
    for i, name in id2label.items():
        lbl = name if isinstance(name, str) else str(name)
        if match_group(lbl, re_car): car_ids.append(int(i))
        if match_group(lbl, re_person): person_ids.append(int(i))
    car_ids = sorted(set(car_ids))
    person_ids = sorted(set(x for x in set(person_ids) if x not in set(car_ids)))
    return car_ids, person_ids

class CarClassificationFundationModel:
    def __init__(self):
        self.preprocessor, self.model, self.car_ids, self.person_ids = self.init_model()

    def init_model(self):
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        id2label = model.config.id2label
        car_ids, person_ids = build_index_sets(id2label)
        return processor, model, car_ids, person_ids

    def predict(self, image_path, as_dict=False):
        image = Image.open(image_path)
        with torch.no_grad():
            inputs = self.preprocessor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

            p_car = probs[self.car_ids].sum().item() if self.car_ids else 0.0
        return easydict.EasyDict({
            'confidence': p_car
        })
    
if __name__ == "__main__":
    model = CarClassificationFundationModel()
    result = model.predict('generated_results/gt/nuscenes_cropped/gen0/0058080_car/361_CAM_BACK.jpg')
    print(result)