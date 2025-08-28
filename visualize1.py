import os 
from visualize.visualize_wireframe import visualize_prediction_comparison, visualize_edge_probabilities
import torch
import matplotlib.pyplot as plt
import numpy as np

def create_individual_visualizations(model, sample_obj, device, output_dir, sample_name):
    """Create comprehensive visualizations for a single sample"""
    print(f"Creating visualizations for {sample_name}...")
    
    # Create sample-specific output directory
    sample_output_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Create prediction comparison
    fig1 = visualize_prediction_comparison(sample_obj, model, device)
    fig1.savefig(os.path.join(sample_output_dir, f'{sample_name}_prediction_comparison.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Get predictions for edge probability visualization
    model.eval()
    with torch.no_grad():
        point_cloud_tensor = torch.FloatTensor(sample_obj.normalized_point_cloud).unsqueeze(0).to(device)
        predictions = model(point_cloud_tensor)
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
    
    # Create edge probability visualization
    fig2 = visualize_edge_probabilities(pred_edge_probs, edge_indices)
    fig2.savefig(os.path.join(sample_output_dir, f'{sample_name}_edge_probabilities.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"✓ Saved visualizations for {sample_name} in {sample_output_dir}")


def main():
    from demo_dataset.PCtoWFdataset import PCtoWFdataset
    from train import evaluate_model
    from evaluate import load_trained_model  # evaluate.py should expose this

    # get model from evaluate.py (no re-training)
    model, dataset, device, train_dataset = load_trained_model()
    model.eval()

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
    
    # Create visualizations for user-selected samples
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAvailable test samples: {len(test_dataset.samples)}")
    print("Enter sample indices to visualize (e.g., '1,3,5' or 'all' for all samples):")
    user_input = input("Sample indices: ").strip()
    
    if user_input.lower() == 'all':
        sample_indices = list(range(len(test_dataset.samples)))
    else:
        try:
            sample_indices = [int(x.strip()) - 1 for x in user_input.split(',') if x.strip()]
            # Validate indices
            sample_indices = [i for i in sample_indices if 0 <= i < len(test_dataset.samples)]
        except ValueError:
            print("Invalid input. Using first sample only.")
            sample_indices = [0]
    
    if not sample_indices:
        print("No valid samples selected. Using first sample.")
        sample_indices = [0]
    
    print(f"Visualizing {len(sample_indices)} sample(s)...")
    for i in sample_indices:
        sample = test_dataset.samples[i]
        create_individual_visualizations(model, sample, device, output_dir, f'test_sample_{i+1}')

if __name__ == "__main__":
    main()
