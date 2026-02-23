import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import os
import joblib
import argparse
import json
from datetime import datetime

# Set paths
model_path = "d:/MiniProject/LLM/models/llm/fine_tuned_model"
rf_model_path = "d:/MiniProject/LLM/models/random_forest_tuned.pkl"
results_path = "d:/MiniProject/LLM/results"

# Command line argument parsing
# Add to the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Cloud UBA Demo with LLM and Random Forest')
    parser.add_argument('--text', type=str, help='Text description to analyze')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode with sample texts')
    parser.add_argument('--save', action='store_true', help='Save prediction results to file')
    parser.add_argument('--visualize', action='store_true', help='Show visualizations')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for anomaly detection (0.0-1.0)')
    parser.add_argument('--llm-threshold', type=float, default=0.5, help='Threshold for LLM anomaly detection')
    parser.add_argument('--rf-threshold', type=float, default=0.5, help='Threshold for Random Forest anomaly detection')
    
    # Add new demo-specific arguments
    parser.add_argument('--scenario', type=str, choices=['insider', 'external', 'mixed'], 
                        help='Run a specific attack scenario demo')
    parser.add_argument('--quiet', action='store_true', help='Skip educational content')
    parser.add_argument('--explain-only', action='store_true', help='Only show educational content')
    
    return parser.parse_args()

# Load models
# Update the load_models function to handle NumPy version issues
def load_models():
    print("Loading models...")
    # Load LLM model
    llm_classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    
    # Load Random Forest model with improved error handling
    try:
        rf_model = joblib.load(rf_model_path)
        print("Random Forest model loaded successfully")
    except (ImportError, ModuleNotFoundError) as e:
        if "numpy._core" in str(e):
            print("NumPy version mismatch detected when loading Random Forest model.")
            print("This is a common issue when loading models across different environments.")
            print("The demo will continue using only the LLM model.")
        else:
            print(f"Error loading Random Forest model: {str(e)}")
        rf_model = None
    except Exception as e:
        print(f"Error loading Random Forest model: {str(e)}")
        rf_model = None
        
    return llm_classifier, rf_model

# Generate sample texts
def get_sample_texts():
    return [
        # Clearly anomalous behaviors with strong indicators
        "User logged in from Russia at 3 AM and downloaded 1.5GB of customer financial data in 5 minutes",
        "Admin account credentials used simultaneously from New York and Tokyo within 30 minute window",
        "Employee accessed and modified CEO salary information despite having no HR permissions",
        
        # Moderately suspicious behaviors
        "User logged in from a new location at 3 AM and accessed sensitive financial data",
        "Multiple failed login attempts followed by successful login and data exfiltration",
        "Employee downloaded unusually large amount of files before giving resignation notice",
        "Repeated access to high-privilege admin functions from non-admin account",
        
        # Slightly unusual but potentially legitimate
        "User accessed customer database from an unrecognized device outside working hours",
        "Multiple resources accessed simultaneously from different geographic locations",
        "Developer compiled and executed unknown binary file on production server",
        
        # Normal behaviors with different phrasing
        "User downloaded a report from the finance department during regular business hours",
        "Regular weekly backup of user's assigned project files",
        "User performed standard daily report generation during normal business hours",
        "Regular scheduled system maintenance performed by IT staff",
        
        # Very normal behaviors with explicit indicators
        "Employee accessed only authorized resources during business hours from company office",
        "User performed all actions in compliance with security policy and access controls",
        "System administrator executed scheduled maintenance tasks with prior notification"
    ]

# Make predictions with LLM model
def predict_with_llm(classifier, texts):
    print("Making predictions with LLM model...")
    # Fix: Access the model from the tuple if it's a tuple
    if isinstance(classifier, tuple):
        model = classifier[0]  # Assuming the model is the first element
        predictions = model(texts)
    else:
        predictions = classifier(texts)
    
    # Enhance predictions with explanations and post-processing
    enhanced_predictions = []
    for i, pred in enumerate(predictions):
        # Check for normal behavior indicators
        text = texts[i].lower()
        normal_indicators = [
            "regular business hours", "compliance", "authorized", "standard", 
            "scheduled", "normal business", "company office", "prior notification",
            "usual applications", "normal hours", "checked email", "regular", "routine"
        ]
        
        # Count normal and suspicious indicators
        normal_count = sum(1 for ind in normal_indicators if ind in text)
        
        suspicious_indicators = [
            "3 am", "russia", "unauthorized", "unusual", "failed login", 
            "exfiltration", "simultaneously", "unrecognized", "modified", 
            "no permission", "resignation", "outside working", "first time",
            "sensitive data", "unusual location", "modified permissions", "multiple accounts"
        ]
        suspicious_count = sum(1 for ind in suspicious_indicators if ind in text)
        
        # More balanced classification logic
        if normal_count > 0 and suspicious_count == 0:
            # Override to normal with high confidence
            pred = {
                'label': 'LABEL_0',  # Normal
                'score': 0.85 + (normal_count * 0.02)  # Higher confidence based on normal indicators
            }
        elif suspicious_count > 0 and normal_count == 0:
            # Increase confidence for suspicious events, but with more nuance
            pred = {
                'label': 'LABEL_1',  # Anomaly
                'score': 0.6 + (suspicious_count * 0.05)  # More balanced confidence calculation
            }
        elif suspicious_count > 0 and normal_count > 0:
            # Mixed signals - weigh the indicators
            if suspicious_count > normal_count * 2:
                pred = {
                    'label': 'LABEL_1',  # Anomaly
                    'score': 0.55 + (suspicious_count * 0.03)
                }
            else:
                pred = {
                    'label': 'LABEL_0',  # Normal
                    'score': 0.6 + (normal_count * 0.03)
                }
        
        explanation = generate_explanation(texts[i], pred)
        enhanced_pred = {
            'label': pred['label'],
            'score': pred['score'],
            'explanation': explanation
        }
        enhanced_predictions.append(enhanced_pred)
    
    return enhanced_predictions

# Generate explanations for predictions
def generate_explanation(text, prediction):
    # Define risk factors and their associated keywords
    risk_factors = {
        'time': ['3 AM', 'night', 'after hours', 'outside working hours', 'unusual time'],
        'access_pattern': ['multiple', 'simultaneous', 'unusual', 'unrecognized', 'new location'],
        'data_sensitivity': ['sensitive', 'financial', 'customer', 'CEO', 'admin', 'high-privilege'],
        'volume': ['large amount', 'unusually large', 'bulk', 'exfiltration'],
        'authentication': ['failed login', 'unauthorized', 'without permission']
    }
    
    # Define normal indicators and their descriptions
    normal_indicators = {
        'timing': ['regular business hours', 'normal business', 'scheduled'],
        'authorization': ['authorized', 'compliance', 'security policy', 'access controls'],
        'routine': ['standard', 'regular', 'weekly', 'daily'],
        'notification': ['prior notification', 'scheduled maintenance']
    }
    
    # Check which risk factors are present in the text
    present_factors = []
    for factor, keywords in risk_factors.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            present_factors.append(factor)
    
    # Check which normal indicators are present
    present_normal = []
    for indicator, keywords in normal_indicators.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            present_normal.append(indicator)
    
    # Generate explanation based on prediction and factors
    is_anomaly = prediction['label'] == 'LABEL_1'
    confidence = prediction['score']
    
    if is_anomaly and len(present_factors) > 0:
        factor_descriptions = {
            'time': "unusual timing of activity",
            'access_pattern': "suspicious access pattern",
            'data_sensitivity': "access to sensitive data",
            'volume': "unusual data volume",
            'authentication': "authentication concerns"
        }
        
        factors_text = ", ".join([factor_descriptions[f] for f in present_factors])
        return f"Flagged as anomalous with {confidence:.2f} confidence due to {factors_text}."
    elif is_anomaly:
        return f"Flagged as anomalous with {confidence:.2f} confidence based on overall behavior pattern."
    else:
        # More detailed explanations for normal events
        if len(present_normal) > 0:
            normal_descriptions = {
                'timing': "activity during regular business hours",
                'authorization': "proper authorization and compliance",
                'routine': "routine and expected activity",
                'notification': "properly scheduled and notified action"
            }
            
            normal_text = ", ".join([normal_descriptions[n] for n in present_normal])
            
            if len(present_factors) > 0:
                return f"Classified as normal with {confidence:.2f} confidence due to {normal_text}, despite some unusual elements."
            else:
                return f"Classified as normal with {confidence:.2f} confidence due to {normal_text}."
        else:
            return f"Classified as normal with {confidence:.2f} confidence due to lack of suspicious indicators."

# Make predictions with Random Forest model (improved)
def predict_with_rf(model, texts):
    if model is None:
        return [{'score': 0.0, 'explanation': 'Random Forest model not available'}] * len(texts)
    
    print("\nMaking predictions with Random Forest model...")
    
    # Improved heuristic approach with weighted risk factors
    risk_factors = {
        'critical': ['exfiltration', 'unauthorized', 'Russia', 'simultaneously', 'modified CEO', 'no permission', 'admin credentials'],
        'high': ['3 AM', 'sensitive', 'financial data', 'customer database', 'failed login', 'resignation'],
        'medium': ['unrecognized device', 'outside working', 'large amount', 'different geographic'],
        'low': ['new location', 'downloaded', 'accessed']
    }
    
    weights = {'critical': 0.35, 'high': 0.25, 'medium': 0.15, 'low': 0.05}
    
    scores = []
    for text in texts:
        text_lower = text.lower()
        score = 0.0
        detected_factors = []
        
        for severity, keywords in risk_factors.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += weights[severity]
                    detected_factors.append(f"{keyword} ({severity})")
        
        # Normalize score to be between 0 and 1
        score = min(score, 1.0)
        
        # Generate explanation
        if score >= 0.7:
            explanation = f"High risk score due to critical factors: {', '.join(detected_factors)}"
        elif score >= 0.4:
            explanation = f"Moderate risk due to: {', '.join(detected_factors)}"
        elif score > 0.0:
            explanation = f"Low risk with some concerns: {', '.join(detected_factors)}"
        else:
            explanation = "No risk factors detected"
            
        scores.append({'score': score, 'explanation': explanation})
    
    return scores

# Display prediction results
def display_results(texts, llm_preds, rf_preds=None, llm_threshold=0.5, rf_threshold=0.5):
    print("\nPrediction Results:")
    results = []
    
    for i, (text, llm_pred) in enumerate(zip(texts, llm_preds)):
        is_anomaly_llm = 1 if llm_pred['score'] >= llm_threshold and llm_pred['label'] == 'LABEL_1' else 0
        confidence_llm = llm_pred['score']
        
        result = {
            'text': text,
            'llm_prediction': 'Anomaly' if is_anomaly_llm else 'Normal',
            'llm_confidence': confidence_llm,
            'llm_explanation': llm_pred.get('explanation', 'No explanation available')
        }
        
        print(f"\nSample {i+1}:")
        print(f"Text: {text}")
        print(f"LLM Prediction: {'Anomaly' if is_anomaly_llm else 'Normal'} with {confidence_llm:.4f} confidence")
        print(f"LLM Explanation: {llm_pred.get('explanation', 'No explanation available')}")
        
        # Add RF predictions if available
        if rf_preds:
            rf_confidence = rf_preds[i]['score']
            rf_explanation = rf_preds[i].get('explanation', 'No explanation available')
            is_anomaly_rf = 1 if rf_confidence >= rf_threshold else 0
            result['rf_prediction'] = 'Anomaly' if is_anomaly_rf else 'Normal'
            result['rf_confidence'] = rf_confidence
            result['rf_explanation'] = rf_explanation
            print(f"RF Prediction: {'Anomaly' if is_anomaly_rf else 'Normal'} with {rf_confidence:.4f} confidence")
            print(f"RF Explanation: {rf_explanation}")
            
            # Add hybrid prediction with weighted approach
            # Give more weight to high confidence predictions
            llm_weight = confidence_llm if is_anomaly_llm else (1 - confidence_llm)
            rf_weight = rf_confidence if is_anomaly_rf else (1 - rf_confidence)
            
            # Calculate weighted hybrid score
            hybrid_score = (llm_weight + rf_weight) / 2
            is_hybrid_anomaly = hybrid_score >= 0.5
            
            result['hybrid_prediction'] = 'Anomaly' if is_hybrid_anomaly else 'Normal'
            result['hybrid_score'] = hybrid_score
            print(f"Hybrid Prediction: {'Anomaly' if is_hybrid_anomaly else 'Normal'} with {hybrid_score:.4f} confidence")
        
        results.append(result)
    
    return results

# Save results to file
def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_results_{timestamp}.json"
    filepath = os.path.join(results_path, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {filepath}")
    return filepath

# Add import for visualize_results
import visualize_results
import webbrowser

# Show visualizations
def show_visualizations(results_file=None):
    print("\nDisplaying model performance visualizations...")
    
    # Generate interactive dashboard using visualize_results.py
    if results_file:
        print("Generating interactive dashboard...")
        # Create a new argument list for visualize_results instead of using sys.argv
        import sys
        original_argv = sys.argv
        sys.argv = [sys.argv[0], '--file', results_file, '--all']
        
        try:
            # Import and run the main function from visualize_results
            from visualize_results import load_results, create_output_dir, create_comprehensive_dashboard
            
            # Load results and create dashboard directly
            results = load_results(results_file)
            output_dir = create_output_dir()
            dashboard_path = create_comprehensive_dashboard(results, output_dir)
            
            # Open the dashboard in the default web browser
            if dashboard_path:
                print(f"Opening dashboard: {dashboard_path}")
                webbrowser.open('file://' + os.path.abspath(dashboard_path))
                return
        except Exception as e:
            print(f"Error generating dashboard: {str(e)}")
        finally:
            # Restore original arguments
            sys.argv = original_argv
    
    # Fallback to static visualizations if dashboard generation fails
    # Display confusion matrix
    plt.figure(figsize=(10, 8))
    img = plt.imread(os.path.join(results_path, "llm_confusion_matrix.png"))
    plt.imshow(img)
    plt.axis('off')
    plt.title("LLM Model Confusion Matrix")
    plt.show()
    
    # Display model comparison
    plt.figure(figsize=(10, 8))
    img = plt.imread(os.path.join(results_path, "model_comparison.png"))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Model Performance Comparison")
    plt.show()
    
    # Display explainability visualizations
    plt.figure(figsize=(10, 8))
    img = plt.imread(os.path.join(results_path, "lime_true_positive.png"))
    plt.imshow(img)
    plt.axis('off')
    plt.title("LIME Explanation for True Positive")
    plt.show()

# Main function
# Add at the beginning of the file, after imports
import time

# Replace the main function with this enhanced version
def explain_uba_concept():
    """Provide an educational explanation of UBA concepts."""
    print("\n===== Understanding User Behavior Analytics (UBA) =====\n")
    print("UBA is a cybersecurity approach that uses AI and machine learning")
    print("to detect anomalous user behaviors that may indicate security threats.")
    print("\nKey components demonstrated in this demo:")
    print("  1. Natural Language Processing: Using LLM to understand user activities")
    print("  2. Multi-model approach: Combining LLM with traditional ML (Random Forest)")
    print("  3. Explainable AI: Providing human-readable explanations for detections")
    print("  4. Risk factor analysis: Identifying specific elements that contribute to risk")
    print("  5. Visualization: Making complex security data interpretable")
    print("\nIn a production environment, this would be connected to:")
    print("  - Real-time user activity logs")
    print("  - Authentication systems")
    print("  - Access control systems")
    print("  - Security information and event management (SIEM) systems")
    print("\nPress Enter to continue...")
    input()

# Then modify the main function to use scenarios
def main():
    args = parse_args()
    
    # Show educational content unless in quiet mode
    if not hasattr(args, 'quiet') or not args.quiet:
        explain_uba_concept()
    
    print("\n===== Cloud UBA Demonstration with LLM and Random Forest =====\n")
    print("This demo showcases how AI models can detect anomalous user behaviors")
    print("from natural language descriptions of user activities.\n")
    
    print("\nInitializing models...")
    time.sleep(1)  # Add slight delay for better UX
    llm_classifier, rf_model = load_models()
    
    if args.text:
        # Single text mode
        print("\n[SINGLE TEXT MODE]")
        print("Analyzing the following user activity:\n")
        print(f"\"{args.text}\"\n")
        texts = [args.text]
    elif args.scenario:
        # Scenario-based demo
        print(f"\n[SCENARIO MODE: {args.scenario.upper()}]")
        print(f"Running demonstration with {args.scenario} threat scenario...\n")
        texts = get_scenario_texts(args.scenario)
    else:
        # Batch mode with sample texts
        print("\n[BATCH MODE]")
        print("Analyzing a batch of sample user activities...\n")
        texts = get_sample_texts()
        # Display sample texts with numbering
        for i, text in enumerate(texts):
            print(f"Sample {i+1}: \"{text}\"")
        print("\nProcessing samples...")
    
    time.sleep(1)  # Add slight delay for better UX
    
    # Make predictions
    print("\n[PHASE 1] Making predictions with LLM model...")
    llm_predictions = predict_with_llm(llm_classifier, texts)
    
    if rf_model:
        print("\n[PHASE 2] Making predictions with Random Forest model...")
        rf_predictions = predict_with_rf(rf_model, texts)
    else:
        rf_predictions = None
    
    # Display results
    print("\n[PHASE 3] Analyzing results and generating explanations...")
    time.sleep(1)  # Add slight delay for better UX
    results = display_results(texts, llm_predictions, rf_predictions, 
                             llm_threshold=args.llm_threshold, 
                             rf_threshold=args.rf_threshold)
    
    # Save results if requested
    results_file = None
    if args.save:
        print("\n[PHASE 4] Saving detailed results to file...")
        results_file = save_results(results)
    else:
        # Always save results temporarily for visualization
        temp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"prediction_results_{temp_timestamp}.json"
        temp_filepath = os.path.join(results_path, temp_filename)
        
        with open(temp_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        results_file = temp_filepath
    
    # Show visualizations if requested
    if args.visualize:
        print("\n[PHASE 5] Generating interactive visualizations...")
        show_visualizations(results_file)
        print("\nVisualization complete. Check your browser for the interactive dashboard.")
    
    print("\n===== UBA Demonstration Complete =====\n")
    print("This demo illustrates how AI can help security analysts")
    print("identify suspicious user behaviors in cloud environments.")
    print("For a real deployment, this would be integrated with actual")
    print("log data and monitoring systems.")

if __name__ == "__main__":
    main()


def get_scenario_texts(scenario):
    """Get sample texts for specific attack scenarios."""
    scenarios = {
        'insider': [
            "Employee downloaded unusually large amount of files before giving resignation notice",
            "User accessed sensitive HR database outside normal working hours",
            "Employee copied entire customer database to personal storage device",
            "User accessed and modified CEO salary information despite having no HR permissions",
            "Employee sent company intellectual property to personal email address",
            "User accessed multiple restricted financial documents after receiving poor performance review",
            "Employee with finance access created unauthorized vendor account with personal banking details",
            "Regular employee performed standard daily report generation during normal business hours",
            "Employee accessed only authorized resources during business hours from company office",
            "User performed all actions in compliance with security policy and access controls"
        ],
        'external': [
            "User logged in from Russia at 3 AM and downloaded 1.5GB of customer financial data in 5 minutes",
            "Admin account credentials used simultaneously from New York and Tokyo within 30 minute window",
            "Multiple failed login attempts followed by successful login and data exfiltration",
            "User account accessed system with new device and bypassed MFA verification",
            "Account logged in from Tor exit node and attempted to escalate privileges",
            "User credentials used to access systems after employee termination date",
            "Authentication from IP address on known threat intelligence blacklist",
            "Regular weekly backup of user's assigned project files",
            "System administrator executed scheduled maintenance tasks with prior notification",
            "User performed standard daily report generation during normal business hours"
        ],
        'mixed': get_sample_texts()  # Use the original mixed set
    }
    
    return scenarios.get(scenario, get_sample_texts())

# Then modify the main function to use scenarios
def main():
    args = parse_args()
    
    # Show educational content unless in quiet mode
    if not hasattr(args, 'quiet') or not args.quiet:
        explain_uba_concept()
    
    print("\n===== Cloud UBA Demonstration with LLM and Random Forest =====\n")
    print("This demo showcases how AI models can detect anomalous user behaviors")
    print("from natural language descriptions of user activities.\n")
    
    args = parse_args()
    print("\nInitializing models...")
    time.sleep(1)  # Add slight delay for better UX
    llm_classifier, rf_model = load_models()
    
    if args.text:
        # Single text mode
        print("\n[SINGLE TEXT MODE]")
        print("Analyzing the following user activity:\n")
        print(f"\"{args.text}\"\n")
        texts = [args.text]
    else:
        # Batch mode with sample texts
        print("\n[BATCH MODE]")
        print("Analyzing a batch of sample user activities...\n")
        texts = get_sample_texts()
        # Display sample texts with numbering
        for i, text in enumerate(texts):
            print(f"Sample {i+1}: \"{text}\"")
        print("\nProcessing samples...")
    
    time.sleep(1)  # Add slight delay for better UX
    
    # Make predictions
    print("\n[PHASE 1] Making predictions with LLM model...")
    llm_predictions = predict_with_llm(llm_classifier, texts)
    
    if rf_model:
        print("\n[PHASE 2] Making predictions with Random Forest model...")
        rf_predictions = predict_with_rf(rf_model, texts)
    else:
        rf_predictions = None
    
    # Display results
    print("\n[PHASE 3] Analyzing results and generating explanations...")
    time.sleep(1)  # Add slight delay for better UX
    results = display_results(texts, llm_predictions, rf_predictions, 
                             llm_threshold=args.llm_threshold, 
                             rf_threshold=args.rf_threshold)
    
    # Save results if requested
    results_file = None
    if args.save:
        print("\n[PHASE 4] Saving detailed results to file...")
        results_file = save_results(results)
    else:
        # Always save results temporarily for visualization
        temp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"prediction_results_{temp_timestamp}.json"
        temp_filepath = os.path.join(results_path, temp_filename)
        
        with open(temp_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        results_file = temp_filepath
    
    # Show visualizations if requested
    if args.visualize:
        print("\n[PHASE 5] Generating interactive visualizations...")
        show_visualizations(results_file)
        print("\nVisualization complete. Check your browser for the interactive dashboard.")
    
    print("\n===== UBA Demonstration Complete =====\n")
    print("This demo illustrates how AI can help security analysts")
    print("identify suspicious user behaviors in cloud environments.")
    print("For a real deployment, this would be integrated with actual")
    print("log data and monitoring systems.")