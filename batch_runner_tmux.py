#!/usr/bin/env python3
"""
Batch job runner that submits jobs 2 at a time using tmux sessions for safety.
Uses tmux sessions to ensure jobs survive terminal disconnections.
"""

import subprocess
import time
from typing import List, Dict, Optional
import argparse
import os
import signal

class TmuxBatchJobRunner:
    def __init__(self, max_concurrent_jobs: int = 2, check_interval: int = 14400):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.check_interval = check_interval
        self.running_jobs: Dict[str, Dict] = {}
        self.completed_jobs: List[str] = []
        self.failed_jobs: List[str] = []
        
    def run_command_async(self, cmd: str, job_name: str) -> subprocess.Popen:
        """Run a command in a tmux session for safer execution."""
        print(f"Starting job: {job_name}")
        
        # Create tmux session for this job
        session_name = f"ml_exp_{job_name}"
        
        # Kill existing session if it exists
        subprocess.run(["tmux", "kill-session", "-t", session_name], 
                      capture_output=True, check=False)
        
        # Create new tmux session and run command
        tmux_cmd = [
            "tmux", "new-session", "-d", "-s", session_name,
            "bash", "-c", f"{cmd}; echo 'EXIT_CODE:'$?; echo 'Job finished. Press Ctrl+C to close session.'; sleep 3600"
        ]
        
        print(f"Tmux session: {session_name}")
        print(f"Command: {cmd}")
        
        process = subprocess.Popen(tmux_cmd)
        
        # Store session info instead of direct process
        self.running_jobs[job_name] = {
            'session': session_name,
            'process': process,
            'start_time': time.time()
        }
        return process
    
    def check_job_status(self, job_name: str) -> Optional[str]:
        """Check if a tmux session job has completed."""
        if job_name not in self.running_jobs:
            return None
            
        job_info = self.running_jobs[job_name]
        session_name = job_info['session']
        
        # Check if tmux session still exists
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True
        )
        
        if result.returncode != 0:
            # Session no longer exists - assume completed
            print(f"✓ Job completed: {job_name} (session ended)")
            self.completed_jobs.append(job_name)
            del self.running_jobs[job_name]
            return 'completed'
        
        # Check if the main process is still running by looking at session content
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", session_name, "-p"],
                capture_output=True, text=True, timeout=5
            )
            
            if "EXIT_CODE:" in result.stdout:
                # Job finished, extract exit code
                lines = result.stdout.strip().split('\n')
                exit_code = None
                for line in lines:
                    if "EXIT_CODE:" in line:
                        exit_code = line.split(':')[-1].strip()
                        break
                
                if exit_code == '0':
                    print(f"✓ Job completed successfully: {job_name}")
                    self.completed_jobs.append(job_name)
                    status = 'completed'
                else:
                    print(f"✗ Job failed: {job_name} (exit code: {exit_code})")
                    self.failed_jobs.append(job_name)
                    status = 'failed'
                
                del self.running_jobs[job_name]
                return status
                
        except subprocess.TimeoutExpired:
            pass
        
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
        print(f"Each job runs in its own tmux session for safety")
        
        for cmd, job_name in commands:
            # Wait for an available slot
            self.wait_for_available_slot()
            
            # Start the job
            self.run_command_async(cmd, job_name)
            
            # Small delay to avoid overwhelming the system
            time.sleep(2)
        
        # Wait for all remaining jobs to complete
        print("Waiting for all jobs to complete...")
        while self.running_jobs:
            for job_name in list(self.running_jobs.keys()):
                self.check_job_status(job_name)
            
            if self.running_jobs:
                print(f"Still running: {list(self.running_jobs.keys())}")
                time.sleep(self.check_interval)
        
        # Print summary
        print(f"\nBatch completed!")
        print(f"Successful jobs: {len(self.completed_jobs)}")
        print(f"Failed jobs: {len(self.failed_jobs)}")
        
        if self.completed_jobs:
            print(f"Successful job names: {', '.join(self.completed_jobs)}")
        
        if self.failed_jobs:
            print(f"Failed job names: {', '.join(self.failed_jobs)}")
        
        # Show how to monitor sessions
        print(f"\nTo monitor running sessions: tmux list-sessions | grep ml_exp")
        print(f"To attach to a session: tmux attach -t ml_exp_<job_name>")
    
    def cleanup_sessions(self):
        """Cleanup any remaining tmux sessions."""
        for job_name, job_info in list(self.running_jobs.items()):
            session_name = job_info['session']
            print(f"Cleaning up session: {session_name}")
            subprocess.run(["tmux", "kill-session", "-t", session_name], 
                         capture_output=True, check=False)
    
    def list_active_sessions(self):
        """List all active ML experiment sessions."""
        result = subprocess.run(["tmux", "list-sessions"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            sessions = [line for line in result.stdout.split('\n') 
                       if 'ml_exp_' in line]
            if sessions:
                print("\\nActive ML experiment sessions:")
                for session in sessions:
                    print(f"  {session}")
            else:
                print("\\nNo active ML experiment sessions found.")
        else:
            print("\\nNo tmux sessions found or tmux not available.")

def create_experiment_commands() -> List[tuple]:
    """Create a list of (command, job_name) tuples for all 12 experiments."""
    base_cmd = "python train_single.py"
    
    experiments = [
        # MOE experiments (1-6)
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
        },
        # Asymmetric experiments (7-12)
        {
            'name': '3q_d512_h4_asym',
            'num_qubits': 3,
            'd_model': 512,
            'nhead': 4,
            'mlp_type': 'asymmetric_ensemble'
        },
        {
            'name': '3q_d512_h8_asym',
            'num_qubits': 3,
            'd_model': 512,
            'nhead': 8,
            'mlp_type': 'asymmetric_ensemble'
        },
        {
            'name': '3q_d1024_h4_asym',
            'num_qubits': 3,
            'd_model': 1024,
            'nhead': 4,
            'mlp_type': 'asymmetric_ensemble'
        },
        {
            'name': '3q_d1024_h8_asym',
            'num_qubits': 3,
            'd_model': 1024,
            'nhead': 8,
            'mlp_type': 'asymmetric_ensemble'
        },
        {
            'name': '3q_d2048_h4_asym',
            'num_qubits': 3,
            'd_model': 2048,
            'nhead': 4,
            'mlp_type': 'asymmetric_ensemble'
        },
        {
            'name': '3q_d2048_h8_asym',
            'num_qubits': 3,
            'd_model': 2048,
            'nhead': 8,
            'mlp_type': 'asymmetric_ensemble'
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
    parser = argparse.ArgumentParser(description='Batch job runner for ML experiments using tmux')
    parser.add_argument('--max-concurrent', type=int, default=2, 
                       help='Maximum number of concurrent jobs (default: 2)')
    parser.add_argument('--check-interval', type=int, default=14400,
                       help='Interval in seconds to check job status (default: 14400 = 4 hours)')
    parser.add_argument('--list-sessions', action='store_true',
                       help='List active ML experiment tmux sessions and exit')
    parser.add_argument('--cleanup', action='store_true',
                       help='Cleanup all ML experiment tmux sessions and exit')
    
    args = parser.parse_args()
    
    # Create the batch runner
    runner = TmuxBatchJobRunner(
        max_concurrent_jobs=args.max_concurrent,
        check_interval=args.check_interval
    )
    
    # Handle utility commands
    if args.list_sessions:
        runner.list_active_sessions()
        return
    
    if args.cleanup:
        print("Cleaning up all ML experiment tmux sessions...")
        result = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            sessions = [s.strip() for s in result.stdout.split('\n') 
                       if s.strip() and 'ml_exp_' in s]
            for session in sessions:
                print(f"Killing session: {session}")
                subprocess.run(["tmux", "kill-session", "-t", session], 
                             capture_output=True, check=False)
            print(f"Cleaned up {len(sessions)} sessions.")
        else:
            print("No tmux sessions found.")
        return
    
    # Get the experiment commands
    commands = create_experiment_commands()
    
    print(f"Prepared {len(commands)} experiments to run")
    print(f"Will run {args.max_concurrent} jobs concurrently")
    print(f"Each job will run in its own tmux session for safety")
    print(f"Check interval: {args.check_interval} seconds ({args.check_interval/3600:.1f} hours)")
    
    # Setup signal handler for graceful cleanup
    def signal_handler(sig, frame):
        print('\\nInterrupted! Cleaning up...')
        runner.cleanup_sessions()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the batch
    runner.run_batch(commands)

if __name__ == "__main__":
    main()