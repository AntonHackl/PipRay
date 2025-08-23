#!/usr/bin/env python3
"""
Script to visualize performance results from multiple JSON files.

This script reads performance timing results from JSON files and creates
a stacked bar chart showing the different phases of the algorithm.
Excludes 'Data Reading' and 'Output' phases to focus on core algorithm performance.
"""

import matplotlib.pyplot as plt
import json
import glob
import os
import numpy as np
from collections import defaultdict


def load_performance_data(filename):
    """Load performance data from a JSON file."""
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {filename}: {e}")
        return None


def extract_phases_from_data(data):
    """Extract phase information from performance data."""
    if data is None or 'phases' not in data:
        return None
    
    phases = {}
    for phase_name, phase_data in data['phases'].items():
        # Skip Data Reading and Output phases
        if phase_name in ['Data Reading', 'Output']:
            continue
        
        # Handle both single run and multi-run formats
        if 'duration_us' in phase_data:
            # Single run format
            phases[phase_name] = phase_data['duration_us']
        elif 'average_us' in phase_data:
            # Multi-run format - use average
            phases[phase_name] = phase_data['average_us']
    
    return phases


def get_all_phase_names(all_data):
    """Get all unique phase names across all datasets, preserving order."""
    phase_names = []
    seen_phases = set()
    for data in all_data:
        if data is not None and 'phases' in data:
            for phase_name in data['phases'].keys():
                # Skip Data Reading and Output phases
                if phase_name in ['Data Reading', 'Output']:
                    continue
                if phase_name not in seen_phases:
                    phase_names.append(phase_name)
                    seen_phases.add(phase_name)
    return phase_names


def create_stacked_bar_chart(json_files, output_file=None):
    """Create a stacked bar chart from multiple JSON performance files."""
    
    # Load all data
    all_data = []
    file_names = []
    
    for json_file in json_files:
        data = load_performance_data(json_file)
        if data is not None:
            all_data.append(data)
            # Extract a meaningful name from the filename, removing 'results_' prefix
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            base_name = base_name.replace('results_', '')
            base_name = base_name.replace('europe_', '')
            file_names.append(base_name)
    
    if not all_data:
        print("No valid data found!")
        return
    
    # Get all unique phase names
    all_phase_names = get_all_phase_names(all_data)
    
    if not all_phase_names:
        print("No valid phases found!")
        return
    
    # Prepare data for plotting
    phase_data = defaultdict(list)
    
    for data in all_data:
        phases = extract_phases_from_data(data)
        if phases:
            for phase_name in all_phase_names:
                phase_data[phase_name].append(phases.get(phase_name, 0))

    for i in range(len(phase_data['Query'])):
        phase_data['Query'][i] = phase_data['Query'][i] / 10
    
    # Convert all phase data from microseconds to milliseconds
    for phase_name in all_phase_names:
        for i in range(len(phase_data[phase_name])):
            phase_data[phase_name][i] = phase_data[phase_name][i] / 1000
    
    # Convert to numpy arrays for easier plotting
    phase_arrays = {}
    for phase_name in all_phase_names:
        phase_arrays[phase_name] = np.array(phase_data[phase_name])
    
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up the bars
    x_positions = np.arange(len(file_names))
    bar_width = 0.6
    
    # Create a color map for phases
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_phase_names)))
    
    # Plot stacked bars
    bottom = np.zeros(len(file_names))
    
    for i, phase_name in enumerate(all_phase_names):
        values = phase_arrays[phase_name]
        ax.bar(x_positions, values, bar_width, bottom=bottom, 
               label=phase_name, color=colors[i], alpha=0.8)
        bottom += values
    
    # Customize the plot
    ax.set_xlabel('Number of Query Points', fontsize=12)
    ax.set_ylabel('Time (milliseconds)', fontsize=12)
    ax.set_title('Performance Comparison on EU parks (subset)', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(file_names, rotation=45, ha='right')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis to show time in appropriate units
    def format_time(x, pos):
        if x >= 1000:
            return f'{x/1000:.1f}s'
        else:
            return f'{x:.0f}ms'
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_time))

    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Performance visualization saved to: {output_file}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n=== Performance Summary ===")
    for phase_name in all_phase_names:
        values = phase_arrays[phase_name]
        if np.any(values > 0):
            avg_time = np.mean(values)
            min_time = np.min(values)
            max_time = np.max(values)
            print(f"{phase_name}:")
            print(f"  Average: {avg_time:.2f} ms")
            print(f"  Range: {min_time:.2f} - {max_time:.2f} ms")
            print()


def find_results_files(pattern="results_*.json"):
    """Find all results JSON files matching the pattern."""
    return sorted(glob.glob(pattern))


def main():
    """Main function."""
    # Hardcoded result JSON files in specific order
    json_files = [
        "results_europe_2000000.json"
    ]
    
    print(f"Processing {len(json_files)} files:")
    for file in json_files:
        print(f"  {file}")
    
    create_stacked_bar_chart(json_files, None)


if __name__ == "__main__":
    main() 