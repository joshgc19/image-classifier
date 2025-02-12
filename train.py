import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import time

from utils.utils import input_args, get_labels, VALIDATE, TEST, TRAIN, LINEAR_SIZE, save_checkpoint


def train_model(data_dir, save_dir, arch, hidden_units, learning_rate, epochs, gpu):
    print("Data directory chosen", data_dir)

    train_dir = data_dir + '/train'
    print("Train Data Directory chosen", train_dir)

    valid_dir = data_dir + '/valid'
    print("Valid Data Directory chosen", train_dir)

    test_dir = data_dir + '/test'
    print("Test Data Directory chosen", train_dir)

    # We define transforms for the training, validation, and testing sets
    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        VALIDATE: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # We load the datasets with ImageFolder
    image_datasets = {
        TRAIN: datasets.ImageFolder(train_dir, transform=data_transforms[TRAIN]),
        VALIDATE: datasets.ImageFolder(valid_dir, transform=data_transforms[VALIDATE]),
        TEST: datasets.ImageFolder(test_dir, transform=data_transforms[TEST])
    }

    # We use the image datasets and the transforms, define the dataloaders
    dataloaders = {
        TRAIN: torch.utils.data.DataLoader(image_datasets[TRAIN], batch_size=64, shuffle=True),
        VALIDATE: torch.utils.data.DataLoader(image_datasets[VALIDATE], batch_size=64, shuffle=True),
        TEST: torch.utils.data.DataLoader(image_datasets[TEST], batch_size=64, shuffle=True),
    }

    # We chose the arch given by the user
    if arch == 'vgg16_bn':
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    elif arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif arch == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("The chosen architecture {} is not valid".format(arch))

    # We turn off training for our parameters to avoid back propagation
    for param in model.parameters():
        param.requires_grad = False

    # We first take the input size of the pretrained model
    model_input = model.classifier[0].in_features

    # We retrieve the labels from the file
    cat_to_name = get_labels()
    # Then we define the amount of labels that we are working with
    num_labels = len(cat_to_name)

    # We create our custom classifier using our layers
    custom_classifier = nn.Sequential(
        nn.Linear(model_input, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, LINEAR_SIZE),
        nn.ReLU(),
        nn.Linear(LINEAR_SIZE, num_labels),
        nn.LogSoftmax(dim=1)
    )

    # At last, we replace our new layers on the pretrained model
    model.classifier = custom_classifier

    # We set up the device using cuda if is available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # We connect the device to the model
    model.to(device)
    print(f'Device used: {device}.\n')

    # Optimizer and Loss function
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    print_every = 20  # Prints results every 20 batches

    validation_losses, training_losses = [], []

    # Start Training
    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0
        batches = 0

        start = time.time()  # Start epoch timer

        # Set model to training mode
        model.train()

        for images, labels in dataloaders[TRAIN]:
            batches += 1
            # Move images and labels to the current device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Resets gradients before forward

            # Forward pass through the model
            model_out = model.forward(images)
            # Loss
            loss = criterion(model_out, labels)
            # Backpropagation
            loss.backward()
            # Adjust parameters based on the backpropagation
            optimizer.step()

            # Metrics
            ps = torch.exp(model_out)
            top_ps, top_class = ps.topk(1, dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            current_accuracy = matches.mean()

            train_loss += loss.item()
            train_accuracy += current_accuracy.item()

            if batches % print_every == 0:
                # Validation phase
                model.eval()  # Freeze weights
                validation_loss = 0
                validation_accuracy = 0
                total_samples = 0

                with torch.no_grad():
                    for images, labels in dataloaders[VALIDATE]:
                        images, labels = images.to(device), labels.to(device)
                        model_out = model(images)
                        loss = criterion(model_out, labels)
                        ps = torch.exp(model_out)
                        top_ps, top_class = ps.topk(1, dim=1)
                        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor).to(device)
                        current_accuracy = matches.mean()

                        validation_loss += loss.item() * images.size(0)  # Scale loss
                        validation_accuracy += current_accuracy.item() * images.size(0)  # Scale accuracy
                        total_samples += images.size(0)

                # Compute final validation metrics
                validation_loss /= total_samples
                validation_accuracy /= total_samples

                # Store losses for analysis
                training_losses.append(train_loss / print_every)
                validation_losses.append(validation_loss)

                # Print Progress
                print('Epoch {}/{} - Processing Batch {}'.format(epoch + 1, epochs, batches))
                print('Average Training Loss (Last {} Batches): {:.3f}'.format(print_every, train_loss / print_every))
                print('Average Training Accuracy (Last {} Batches): {:.2f}%'.format(print_every,
                                                                                    train_accuracy / print_every * 100))
                print('Validation Loss: {:.3f} (Across {} Batches)'.format(validation_loss / len(dataloaders[VALIDATE]),
                                                                           len(dataloaders[VALIDATE])))
                print('Validation Accuracy: {:.2f}% (Across {} Batches)'.format(validation_accuracy * 100, len(dataloaders[VALIDATE])))

                train_loss = train_accuracy = 0
                model.train()

        end = time.time()  # End epoch timer
        print(f"Epoch {epoch + 1} completed in {end - start:.2f} seconds")

    # Initialize test accuracy
    total_accuracy = 0

    # Record start time for validation
    start_time = time.time()
    print('Validation started.')

    # Loop over the test dataset
    for images, labels in dataloaders[TEST]:
        model.eval()  # Set model to evaluation mode
        images, labels = images.to(device), labels.to(device)  # Move images and labels to the device

        # Perform forward pass
        log_probs = model(images)  # Log probabilities from the model
        probs = torch.exp(log_probs)  # Convert log probs to probabilities

        # Get top predictions and corresponding classes
        top_probs, top_classes = probs.topk(1, dim=1)

        # Compare predictions with ground truth labels
        correct_matches = (top_classes == labels.view(*top_classes.shape)).float()

        # Calculate accuracy for the current batch
        batch_accuracy = correct_matches.mean()
        total_accuracy += batch_accuracy

    # Record end time for validation
    end_time = time.time()
    print('Validation ended.')

    # Calculate and print total validation time
    validation_duration = end_time - start_time
    print(f'Validation time: {validation_duration // 60:.0f}m {validation_duration % 60:.0f}s')

    # Print final test accuracy
    test_accuracy = total_accuracy / len(dataloaders[TEST])  # Average over all batches
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Mapping from class labels to indices
    class_to_index = image_datasets[TRAIN].class_to_idx

    # Save the model checkpoint
    save_checkpoint(model, hidden_units, num_labels, save_dir, arch, class_to_index)


if __name__ == "__main__":
    args = input_args()
    train_model(args.data_dir, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)