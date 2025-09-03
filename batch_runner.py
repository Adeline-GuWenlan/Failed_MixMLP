#!/usr/bin/env python3
"""
Batch job runner that submits 2 jobs at a time and waits for completion
before submitting the next batch.
"""

import subprocess
import time
import sys
import threading
from typing import List, Dict, Optional
import argparse

class BatchJobRunner:
    def __init__(self, max_concurrent_jobs: int = 2, check_interval: int = 30):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.check_interval = check_interval
        self.running_jobs: Dict[str, subprocess.Popen] = {}
        self.completed_jobs: List[str] = []
        self.failed_jobs: List[str] = []
        
    def run_command_async(self, cmd: str, job_name: str) -> subprocess.Popen:
        """Run a command asynchronously and return the process."""
        print(f"Starting job: {job_name}")
        print(f"Command: {cmd}")
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.running_jobs[job_name] = process
        return process
    
    def check_job_status(self, job_name: str) -> Optional[str]:
        """Check if a job has completed. Returns 'completed', 'failed', or None if still running."""
        if job_name not in self.running_jobs:
            return None
            
        process = self.running_jobs[job_name]
        return_code = process.poll()
        
        if return_code is not None:
            # Job has completed
            stdout, stderr = process.communicate()
            
            if return_code == 0:
                print(f"✓ Job completed successfully: {job_name}")
                self.completed_jobs.append(job_name)
                del self.running_jobs[job_name]
                return 'completed'
            else:
                print(f"✗ Job failed: {job_name} (exit code: {return_code})")
                print(f"Error output: {stderr}")
                self.failed_jobs.append(job_name)
                del self.running_jobs[job_name]
                return 'failed'
        
        return None
    
    def wait_for_available_slot(self):
        """Wait until there's an available slot for a new job."""
        while len(self.running_jobs) >= self.max_concurrent_jobs:
            print(f"Waiting for jobs to complete... ({len(self.running_jobs)} running)")
            
            # Check status of all running jobs
            completed_jobs = []
            for job_name in list(self.running_jobs.keys()):
                status = self.check_job_status(job_name)
                if status is not None:
                    completed_jobs.append(job_name)
            
            if not completed_jobs:
                time.sleep(self.check_interval)
    
    def run_batch(self, commands: List[tuple]):
        """Run a batch of commands with the specified concurrency limit."""
        print(f"Starting batch run with {len(commands)} jobs, max {self.max_concurrent_jobs} concurrent")
        
        for cmd, job_name in commands:
            # Wait for an available slot
            self.wait_for_available_slot()
            
            # Start the job
            self.run_command_async(cmd, job_name)
            
            # Small delay to avoid overwhelming the system
            time.sleep(1)
        
        # Wait for all remaining jobs to complete
        print("Waiting for all jobs to complete...")
        while self.running_jobs:
            for job_name in list(self.running_jobs.keys()):
                self.check_job_status(job_name)
            time.sleep(self.check_interval)
        
        # Print summary
        print(f"\nBatch completed!")
        print(f"Successful jobs: {len(self.completed_jobs)}")
        print(f"Failed jobs: {len(self.failed_jobs)}")
        
        if self.failed_jobs:
            print(f"Failed job names: {', '.join(self.failed_jobs)}")

def create_experiment_commands() -> List[tuple]:
    """Create a list of (command, job_name) tuples for the experiments."""
    base_cmd = "python train_single.py"
    
    experiments = [
        {
            'name': '3q_d512_h4_moe',
            'num_qubits': 3,
            'd_model': 512,
            'nhead': 4,
            'mlp_type': 'mixture_of_experts'
        },
        {
            'name': '3q_d512_h8_moe', 
            'num_qubits': 3,
            'd_model': 512,
            'nhead': 8,
            'mlp_type': 'mixture_of_experts'
        },
        {
            'name': '3q_d1024_h4_moe',
            'num_qubits': 3,
            'd_model': 1024,
            'nhead': 4,
            'mlp_type': 'mixture_of_experts'
        },
        {
            'name': '3q_d1024_h8_moe',
            'num_qubits': 3,
            'd_model': 1024,
            'nhead': 8,
            'mlp_type': 'mixture_of_experts'
        },
        {
            'name': '3q_d2048_h4_moe',
            'num_qubits': 3,
            'd_model': 2048,
            'nhead': 4,
            'mlp_type': 'mixture_of_experts'
        },
        {
            'name': '3q_d2048_h8_moe',
            'num_qubits': 3,
            'd_model': 2048,
            'nhead': 8,
            'mlp_type': 'mixture_of_experts'
        }
    ]
    
    commands = []
    for exp in experiments:
        cmd = f"""{base_cmd} --exp_name {exp['name']} --num_qubits {exp['num_qubits']} --d_model {exp['d_model']} --nhead {exp['nhead']} \
--pooling_type cls --mlp_type {exp['mlp_type']} --use_physics_mask False --mask_threshold 1.0 \
--use_cls_token True --epochs 100 --lr 1e-4 --batch_size 512 \
--data_folder Decoded_Tokens \
--labels_path Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy \
--train_ratio 0.8 --val_ratio 0.1 --num_workers 4 \
--device auto --base_log_dir experiments --base_model_dir models"""
        
        commands.append((cmd, exp['name']))
    
    return commands

def main():
    parser = argparse.ArgumentParser(description='Batch job runner for ML experiments')
    parser.add_argument('--max-concurrent', type=int, default=2, 
                       help='Maximum number of concurrent jobs (default: 2)')
    parser.add_argument('--check-interval', type=int, default=14400,
                       help='Interval in seconds to check job status (default: 14400 = 4 hours)')
    
    args = parser.parse_args()
    
    # Create the batch runner
    runner = BatchJobRunner(
        max_concurrent_jobs=args.max_concurrent,
        check_interval=args.check_interval
    )
    
    # Get the experiment commands
    commands = create_experiment_commands()
    
    print(f"Prepared {len(commands)} experiments to run")
    print(f"Will run {args.max_concurrent} jobs concurrently")
    
    # Run the batch
    runner.run_batch(commands)

if __name__ == "__main__":
    main()