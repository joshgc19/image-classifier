
import torch
from torchvision import models
from utils.utils import load_model_checkpoint, process_image, get_args, get_labels

def predict(image_path, checkpoint_path, top_k, gpu):
    arch, input_size, output_size, hidden_units, state_dict, class_to_idx, classifier = load_model_checkpoint(checkpoint_path, gpu)
    model = getattr(models, arch)(pretrained=True)
    # Load the model state_dict into the model
    model.classifier =  classifier
    model.load_state_dict(state_dict)
    model.class_to_idx = class_to_idx

    # We set up the device using cuda if is available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    print(f'Device used: {device}.\n')

    processed_image = torch.from_numpy(process_image(image_path)).unsqueeze(0).to(device).float()

    model.to(device)
    model.eval()

    with torch.no_grad():  # Avoid gradient computation during inference
        log_ps = model(processed_image)
        ps = torch.exp(log_ps)  # Convert log probabilities to actual probabilities

    top_ps, top_idx = ps.topk(top_k, dim=1)

    probabilities = top_ps.tolist()[0]
    indices = top_idx.tolist()[0]

    # Map indices to class names
    classes = [class_to_idx[idx] for idx in indices]

    return probabilities, classes

if __name__ == "__main__":
    args = get_args()
    top_probs, top_classes = predict(args.image_path, args.checkpoint, args.top_k, args.gpu)

    #Category names
    cat_to_name = get_labels(labels_path=args.cat_to_name)

    category_names = [cat_to_name[class_] for class_ in top_classes]

    #Print results
    for prob, class_, category_name in zip(top_probs, top_classes, category_names):
        print(f"Class: {class_}, Category: {category_name}, Probability: {prob:.4f}")