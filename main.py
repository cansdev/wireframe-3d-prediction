import torch
from demo_dataset.PCtoWFdataset import PCtoWFdataset
from train import evaluate_model, train_overfit_model

def main():

    # Load data
    dataset = PCtoWFdataset( 
        train_pc_dir='demo_dataset/train_dataset/point_cloud',
        train_wf_dir='demo_dataset/train_dataset/wireframe',
        test_pc_dir='demo_dataset/test_dataset/point_cloud',
        test_wf_dir='demo_dataset/test_dataset/wireframe'
    )
    
    # Print dataset info
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("STARTING BATCH TRAINING")
    print("="*60)

    # Load training dataset with multiple files
    # Initializes sample object
    train_dataset = dataset.load_training_dataset()
    
    # Load and preprocess all training data at once
    # Creates sample object, loads data, creates adjacency matrix, normalizes data
    train_dataset.load_all_data()

    # Gets batch data
    batch_data = train_dataset.get_batch_data(target_points=1024)

    model, loss_history = train_overfit_model(batch_data, num_epochs=1000, learning_rate=0.001)

    print("\n" + "="*50)
    print("EVALUATING TRAINED MODEL")
    print("="*50)
    
    # Get max vertices for evaluation
    max_vertices = train_dataset.max_vertices
    
    # Load and evaluate test dataset
    test_dataset = dataset.load_testing_dataset()
    test_dataset.load_all_data()
    test_batch_data = test_dataset.get_batch_data(target_points=1024)
    
    test_results = evaluate_model(model, test_batch_data, device, max_vertices)
    
    # Print results
    print("\nTest Results:")
    print("-" * 40)
    for result in test_results:
        print(f"Sample {result['sample_index']+1}:")
        print(f"  Vertex RMSE: {result['vertex_rmse']:.6f}")
        print(f"  Edge Accuracy: {result['edge_accuracy']:.6f}")
        print(f"  Edge Precision: {result['edge_precision']:.6f}")
        print(f"  Edge Recall: {result['edge_recall']:.6f}")
        print(f"  Edge F1-Score: {result['edge_f1_score']:.6f}")
        print()
    
    # Save model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("\nModel saved as 'trained_model.pth'")
    
    print("\nOvertraining completed successfully!")

if __name__ == "__main__":
    main()
