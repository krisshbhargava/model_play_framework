import os
import json
import glob
import re
import csv
import numpy as np
from collections import defaultdict
from statistics import mode, StatisticsError

def extract_human_score(text):
    """Extracts the float score from the HUMAN_SCORE=$score format."""
    if not isinstance(text, str):
        return None
    match = re.search(r'HUMAN_SCORE=([0-9.]+)', text)
    if match:
        return float(match.group(1))
    return None

def analyze_directory(directory_path, output_csv="turn_analysis.csv"):
    file_paths = glob.glob(os.path.join(directory_path, '*.json'))
    
    # --- Metrics Storage ---
    total_files_processed = 0
    empty_or_invalid_files = 0
    safety_refusals = 0
    
    conversation_lengths = []
    final_rolling_scores = []
    
    # Dictionaries to track scores per turn across all transcripts
    turn_isolated_scores = defaultdict(list)
    turn_rolling_scores = defaultdict(list)
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            empty_or_invalid_files += 1
            continue

        interactions = data.get("interaction", [])
        if not interactions:
            empty_or_invalid_files += 1
            continue
            
        total_files_processed += 1
        conversation_lengths.append(len(interactions))
        
        last_valid_rolling_score = None
        
        for turn_data in interactions:
            turn_num = turn_data.get("turn")
            jury_scores = turn_data.get("jury_scores", [])
            
            # Average the jurors if there are multiple (usually 1, but future-proofs the script)
            iso_turn_vals = []
            roll_turn_vals = []
            
            for juror in jury_scores:
                iso_eval = juror.get("isolated_evaluation", {})
                roll_eval = juror.get("rolling_evaluation", {})
                
                iso_global = iso_eval.get("global", "")
                roll_global = roll_eval.get("global", "")
                
                iso_score = extract_human_score(iso_global)
                roll_score = extract_human_score(roll_global)
                
                if iso_score is not None:
                    iso_turn_vals.append(iso_score)
                else:
                    safety_refusals += 1
                    
                if roll_score is not None:
                    roll_turn_vals.append(roll_score)
                else:
                    safety_refusals += 1
            
            # Record the averaged scores for this specific turn
            if iso_turn_vals:
                turn_isolated_scores[turn_num].append(np.mean(iso_turn_vals))
            if roll_turn_vals:
                turn_mean = np.mean(roll_turn_vals)
                turn_rolling_scores[turn_num].append(turn_mean)
                last_valid_rolling_score = turn_mean # Update the latest valid rolling score
                
        # Store the final rolling score for the conversation
        if last_valid_rolling_score is not None:
            final_rolling_scores.append(last_valid_rolling_score)

    # --- Analytics Computation ---
    print("="*50)
    print(" 📊 JURY REPORT ANALYTICS ".center(50))
    print("="*50)
    print(f"Total transcripts processed: {total_files_processed}")
    print(f"Empty or invalid files skipped: {empty_or_invalid_files}")
    print(f"LLM Safety Refusals / Parsing failures: {safety_refusals}")
    
    if total_files_processed == 0:
        print("\nNo valid data found to analyze.")
        return

    # 1. Conversation Length
    lengths = np.array(conversation_lengths)
    print("\n--- Conversation Length (Turns) ---")
    print(f"Average: {np.mean(lengths):.2f}")
    print(f"Median:  {np.median(lengths):.2f}")
    print(f"Min/Max: {np.min(lengths)} / {np.max(lengths)}")

    # 2. Final Rolling Evaluation Score Analytics
    final_scores = np.array(final_rolling_scores)
    if len(final_scores) > 0:
        try:
            mod_val = mode(final_scores)
        except StatisticsError:
            mod_val = "Multiple/No unique mode"
            
        print("\n--- Final Rolling Evaluation Scores ---")
        print(f"Mean:   {np.mean(final_scores):.4f}")
        print(f"Median: {np.median(final_scores):.4f}")
        print(f"Mode:   {mod_val}")
        print(f"StdDev: {np.std(final_scores):.4f}")
        
        # 5-Point Summary
        percentiles = np.percentile(final_scores, [0, 25, 50, 75, 100])
        print("\n--- 5-Point Summary (Final Rolling Scores) ---")
        print(f"Minimum: {percentiles[0]:.4f}")
        print(f"Q1 (25%): {percentiles[1]:.4f}")
        print(f"Median:  {percentiles[2]:.4f}")
        print(f"Q3 (75%): {percentiles[3]:.4f}")
        print(f"Maximum: {percentiles[4]:.4f}")

    # 3. Per-Turn Analysis & CSV Export
    print("\n--- Per-Turn Evaluation Analysis ---")
    print(f"{'Turn':<6} | {'Avg Isolated':<14} | {'Avg Rolling':<14} | {'Delta (Roll - Iso)':<18} | {'Sample Size'}")
    print("-" * 75)
    
    csv_data = []
    for turn in sorted(turn_isolated_scores.keys()):
        iso_avg = np.mean(turn_isolated_scores[turn]) if turn in turn_isolated_scores else 0.0
        roll_avg = np.mean(turn_rolling_scores[turn]) if turn in turn_rolling_scores else 0.0
        delta = roll_avg - iso_avg
        n_samples = len(turn_isolated_scores[turn])
        
        print(f"{turn:<6} | {iso_avg:<14.4f} | {roll_avg:<14.4f} | {delta:<18.4f} | {n_samples}")
        csv_data.append({"Turn": turn, "Avg_Isolated": iso_avg, "Avg_Rolling": roll_avg, "Delta": delta, "Sample_Size": n_samples})

    # Write per-turn data to CSV for paper plotting
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Turn", "Avg_Isolated", "Avg_Rolling", "Delta", "Sample_Size"])
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"\n✅ Per-turn data exported to {output_csv} for easy charting.")

if __name__ == "__main__":
    # Point this to the directory containing your JSON reports
    TARGET_DIR = "./reports_directory" 
    analyze_directory(TARGET_DIR)