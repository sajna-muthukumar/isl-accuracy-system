import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_model(n):
    return nn.Sequential(
        nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64*16*16,128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128,n)
    )

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="data"); args=ap.parse_args()
    ckpt_path=Path("artifacts/best_model.pt"); classes_path=Path("artifacts/classes.json")
    if not ckpt_path.exists() or not classes_path.exists():
        raise SystemExit("Train the model first using train.py.")
    classes=json.loads(classes_path.read_text(encoding="utf-8"))
    test_dir=Path(args.data)/"test"
    if not test_dir.is_dir(): raise SystemExit("Expected data/test/<class>/ folders.")
    tf=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),
                           transforms.Normalize((0.5,)*3,(0.5,)*3)])
    ds=datasets.ImageFolder(test_dir,transform=tf)
    if ds.classes != classes: raise SystemExit(f"Test classes {ds.classes} do not match trained classes {classes}")
    dl=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(len(classes)).to(device)
    checkpoint=torch.load(ckpt_path,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in dl:
            pred=model(x.to(device)).argmax(1).cpu().tolist()
            y_pred.extend(pred); y_true.extend(y.tolist())
    print(f"Test images: {len(y_true)}")
    print(f"Test accuracy: {accuracy_score(y_true,y_pred):.4f}")
    print("\\nClassification report:")
    print(classification_report(y_true,y_pred,target_names=classes,zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true,y_pred))
if __name__=="__main__": main()
