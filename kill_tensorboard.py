#!/usr/bin/env python3
"""
Kill existing TensorBoard processes and check what they were running with
"""

import subprocess
import os
import signal


def kill_tensorboard_processes():
    """Find and kill existing TensorBoard processes"""
    
    print("🔍 Checking for running TensorBoard processes...\n")
    
    try:
        # Find TensorBoard processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        tensorboard_processes = []
        
        for line in result.stdout.split('\n'):
            if 'tensorboard' in line and 'python' in line:
                tensorboard_processes.append(line)
        
        if tensorboard_processes:
            print(f"Found {len(tensorboard_processes)} TensorBoard process(es):")
            for i, proc in enumerate(tensorboard_processes):
                print(f"  {i+1}. {proc}")
                
                # Try to extract the logdir argument
                if '--logdir' in proc:
                    parts = proc.split('--logdir')
                    if len(parts) > 1:
                        logdir_part = parts[1].split()[0]
                        print(f"     → Using logdir: {logdir_part}")
            
            print()
            
            # Kill the processes
            for proc in tensorboard_processes:
                try:
                    # Extract PID (second column in ps aux output)
                    pid = int(proc.split()[1])
                    print(f"Killing TensorBoard process {pid}...")
                    os.kill(pid, signal.SIGTERM)
                    print(f"✅ Killed process {pid}")
                except Exception as e:
                    print(f"❌ Failed to kill process: {e}")
            
            print("\n✅ All TensorBoard processes terminated")
            print("Now you can restart training and TensorBoard should use the correct directory")
            
        else:
            print("❌ No running TensorBoard processes found")
            print("The issue might be something else...")
            
    except Exception as e:
        print(f"Error checking processes: {e}")


def check_tensorboard_port():
    """Check what's running on port 6006"""
    
    print("\n🔍 Checking what's running on port 6006...\n")
    
    try:
        result = subprocess.run(['lsof', '-i', ':6006'], capture_output=True, text=True)
        if result.stdout:
            print("Port 6006 is being used by:")
            print(result.stdout)
        else:
            print("Port 6006 is not in use")
    except Exception as e:
        print(f"Could not check port 6006: {e}")


if __name__ == '__main__':
    print("🚀 TensorBoard Process Manager\n")
    
    kill_tensorboard_processes()
    check_tensorboard_port()
    
    print("\n💡 After killing processes:")
    print("1. Start training again")
    print("2. Check the debug logs to see what directory TensorBoard is started with")
    print("3. The TensorBoard tab should refresh and show the correct data")
