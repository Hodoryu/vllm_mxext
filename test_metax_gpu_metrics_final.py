#!/usr/bin/env python3
"""
Final fixed test script for collect_metax_gpu_metrics function.
"""

import subprocess
import json
from typing import Dict, List

def collect_metax_gpu_metrics() -> List[Dict]:
    """Collect MetaX GPU metrics using mx-smi command."""
    print("=" * 60)
    print("Starting MetaX GPU metrics collection test...")
    print("=" * 60)
    
    gpu_metrics = []
    
    try:
        print("Step 1: Executing mx-smi command...")
        result = subprocess.run(['mx-smi'], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"❌ ERROR: mx-smi command failed with return code {result.returncode}")
            return gpu_metrics
        
        print("✅ mx-smi command executed successfully")
        
        output = result.stdout
        lines = output.split('\n')
        
        print(f"\nStep 2: Parsing GPU information...")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for GPU info lines that start with "| <gpu_id>"
            if line.startswith('|') and len(line.split('|')) >= 4:
                parts = [part.strip() for part in line.split('|') if part.strip()]
                
                # Check if this is a GPU info line by looking for GPU ID at the start
                if len(parts) >= 3:
                    # Extract GPU ID from the first part (e.g., "0           MetaX C280")
                    first_part = parts[0].strip()
                    print(f"\n  Processing potential GPU line {i}: '{line}'")
                    print(f"    First part: '{first_part}'")
                    
                    if first_part:  # Make sure it's not empty
                        gpu_id_match = first_part.split()[0]  # Get the first word
                        print(f"    First word: '{gpu_id_match}'")
                        
                        # Check if the first word is a digit (GPU ID)
                        if gpu_id_match.isdigit():
                            print(f"    ✅ Found GPU ID: {gpu_id_match}")
                            try:
                                gpu_id = int(gpu_id_match)
                                
                                # Extract GPU name (everything after the GPU ID)
                                gpu_name_parts = first_part.split()[1:]  # Skip the first word (GPU ID)
                                gpu_name = ' '.join(gpu_name_parts)  # Join the rest as GPU name
                                
                                bus_id = parts[1].strip()    # e.g., "0000:17:00.0"
                                gpu_util_str = parts[2].strip()  # e.g., "0%"
                                gpu_util = int(gpu_util_str.replace('%', ''))
                                
                                print(f"    - GPU ID: {gpu_id}")
                                print(f"    - GPU Name: '{gpu_name}'")
                                print(f"    - Bus ID: {bus_id}")
                                print(f"    - GPU Utilization: {gpu_util}%")
                                
                                # Get the next line for temperature, power, and memory info
                                if i + 1 < len(lines):
                                    next_line = lines[i + 1].strip()
                                    print(f"    Next line {i+1}: '{next_line}'")
                                    next_parts = [part.strip() for part in next_line.split('|') if part.strip()]
                                    print(f"    Next line parts: {next_parts}")
                                    
                                    if len(next_parts) >= 2:
                                        # Parse temperature and power from first part (e.g., "40C         73W / 280W")
                                        temp_power_str = next_parts[0].strip()
                                        print(f"    Temp/Power string: '{temp_power_str}'")
                                        temp_power_parts = temp_power_str.split()
                                        print(f"    Temp/Power parts: {temp_power_parts}")
                                        
                                        # Extract temperature (e.g., "40C")
                                        temperature = int(temp_power_parts[0].replace('C', ''))
                                        
                                        # Extract power (e.g., "73W / 280W")
                                        power_str = ' '.join(temp_power_parts[1:])  # "73W / 280W"
                                        print(f"    Power string: '{power_str}'")
                                        power_parts = power_str.split('/')
                                        power_usage = int(power_parts[0].strip().replace('W', ''))
                                        power_limit = int(power_parts[1].strip().replace('W', ''))
                                        
                                        # Parse memory (e.g., "54995/65536 MiB")
                                        memory_str = next_parts[1].strip()
                                        print(f"    Memory string: '{memory_str}'")
                                        memory_parts = memory_str.replace('MiB', '').split('/')
                                        memory_used_mib = int(memory_parts[0].strip())
                                        memory_total_mib = int(memory_parts[1].strip())
                                        
                                        print(f"    - Temperature: {temperature}°C")
                                        print(f"    - Power: {power_usage}W / {power_limit}W")
                                        print(f"    - Memory: {memory_used_mib}/{memory_total_mib} MiB")
                                        
                                        # Calculate derived values
                                        memory_percent = (memory_used_mib / memory_total_mib) * 100 if memory_total_mib > 0 else 0
                                        power_percent = (power_usage / power_limit) * 100 if power_limit > 0 else 0
                                        
                                        gpu_metric = {
                                            'id': gpu_id,
                                            'name': gpu_name,
                                            'load': gpu_util,
                                            'memory_used': memory_used_mib,
                                            'memory_total': memory_total_mib,
                                            'memory_percent': memory_percent,
                                            'temperature': temperature,
                                            'power_usage': power_usage,
                                            'power_limit': power_limit,
                                            'power_percent': power_percent,
                                            'bus_id': bus_id
                                        }
                                        
                                        gpu_metrics.append(gpu_metric)
                                        print(f"    ✅ Successfully processed GPU {gpu_id}")
                                        
                            except (ValueError, IndexError) as e:
                                print(f"    ❌ Error parsing GPU line: {line}, error: {e}")
                        else:
                            print(f"    ❌ First word '{gpu_id_match}' is not a digit")
                            
            i += 1
            
        print(f"\nStep 3: Final results")
        print(f"  - Total GPUs processed: {len(gpu_metrics)}")
        
        return gpu_metrics
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return gpu_metrics


def main():
    """Main test function."""
    print("MetaX GPU Metrics Collection Test (Final Fixed Version)")
    print("=" * 60)
    
    gpu_metrics = collect_metax_gpu_metrics()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if gpu_metrics:
        print(f"✅ SUCCESS: Found {len(gpu_metrics)} GPU(s)")
        print("\nDetailed GPU Information:")
        
        for i, gpu in enumerate(gpu_metrics):
            print(f"\n--- GPU {i+1} ---")
            print(json.dumps(gpu, indent=2))
        
        print(f"\n📊 Summary Statistics:")
        avg_load = sum(gpu['load'] for gpu in gpu_metrics) / len(gpu_metrics)
        avg_temp = sum(gpu['temperature'] for gpu in gpu_metrics) / len(gpu_metrics)
        avg_memory = sum(gpu['memory_percent'] for gpu in gpu_metrics) / len(gpu_metrics)
        avg_power = sum(gpu['power_percent'] for gpu in gpu_metrics) / len(gpu_metrics)
        
        print(f"  - Average GPU Load: {avg_load:.1f}%")
        print(f"  - Average Temperature: {avg_temp:.1f}°C")
        print(f"  - Average Memory Usage: {avg_memory:.1f}%")
        print(f"  - Average Power Usage: {avg_power:.1f}%")
        
        print(f"\n🎯 Function is working correctly!")
        
    else:
        print("❌ FAILURE: No GPU metrics collected")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()