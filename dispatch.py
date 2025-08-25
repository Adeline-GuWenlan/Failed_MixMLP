#!/usr/bin/env python3
"""
Enhanced Distributed Experiment Launcher
Fixes parameter mismatches and adds flexible experiment configuration.
Supports both predefined experiments and grid search.
"""

import os
import subprocess
import time
import argparse
import json
import itertools
from datetime import datetime
from pathlib import Path


class ExperimentDispatcher:
    """Manages distributed experiment launching via tmux"""
    
    def __init__(self, log_dir="logs", dry_run=False, verbose=True):
        self.log_dir = Path(log_dir)
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.log_dir / "stdout").mkdir(exist_ok=True)
        (self.log_dir / "stderr").mkdir(exist_ok=True)
    
    def check_tmux_available(self):
        """Check if tmux is available"""
        try:
            subprocess.run(["tmux", "-V"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_session_exists(self, session_name):
        """Check if tmux session already exists"""
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", session_name], 
                capture_output=True, 
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def kill_session(self, session_name):
        """Kill a tmux session if it exists"""
        if self.check_session_exists(session_name):
            cmd = ["tmux", "kill-session", "-t", session_name]
            if not self.dry_run:
                subprocess.run(cmd, capture_output=True)
            if self.verbose:
                print(f"🗑️  Killed existing session: {session_name}")

    def launch_experiment(self, exp_config, base_args, device_id=None, delay=0):
        """Launch a single experiment in a tmux session - ENHANCED VERSION"""
        exp_name = exp_config["name"]
        
        # Kill existing session if it exists
        self.kill_session(exp_name)
        
        # Combine base args with experiment-specific parameters
        cmd_args = {**base_args}
        
        # Add experiment-specific parameters
        if "params" in exp_config:
            cmd_args.update(exp_config["params"])
        
        # Add device assignment
        if device_id is not None:
            cmd_args["device"] = f"cuda:{device_id}" if device_id >= 0 else "cpu"
        
        # Add experiment name
        cmd_args["exp_name"] = exp_name
        
        # Use specific Python environment (modify path as needed)
        python_path = os.path.expanduser("~/miniconda3/envs/magic/bin/python")
        
        # Create command with all parameters
        train_cmd = self.create_experiment_command(cmd_args, python_path=python_path)
        
        # Setup logging paths
        stdout_log = self.log_dir / "stdout" / f"{exp_name}.log"
        stderr_log = self.log_dir / "stderr" / f"{exp_name}_err.log"
        
        # Create full command with logging
        full_cmd = f"{train_cmd} > {stdout_log} 2> {stderr_log}"
        
        if self.verbose:
            print(f"🚀 Launching: {exp_name}")
            if device_id is not None:
                print(f"   Device: {'cuda:'+str(device_id) if device_id >= 0 else 'cpu'}")
            print(f"   Python: {python_path}")
            print(f"   Command: {train_cmd}")
            print(f"   Logs: {stdout_log} | {stderr_log}")
        
        if not self.dry_run:
            # Add delay before launching
            if delay > 0:
                time.sleep(delay)
            
            try:
                # Create empty tmux session
                subprocess.run(["tmux", "new-session", "-d", "-s", exp_name], check=True, capture_output=True)
                
                # Send the command to the session
                subprocess.run(["tmux", "send-keys", "-t", exp_name, full_cmd, "Enter"], check=True, capture_output=True)
                
                # Send exit command so session closes when done
                subprocess.run(["tmux", "send-keys", "-t", exp_name, "exit", "Enter"], check=True, capture_output=True)
                
                if self.verbose:
                    print(f"✅ Started session: {exp_name}")
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to start {exp_name}: {e}")
                if e.stdout:
                    print(f"   stdout: {e.stdout.decode()}")
                if e.stderr:
                    print(f"   stderr: {e.stderr.decode()}")
                return False
        else:
            print(f"[DRY RUN] Would run:")
            print(f"   tmux new-session -d -s {exp_name}")
            print(f"   tmux send-keys -t {exp_name} '{full_cmd}' Enter")
            print(f"   tmux send-keys -t {exp_name} 'exit' Enter")
        
        return True

    def create_experiment_command(self, cmd_args, python_path="python"):
        """Create the command to run a single experiment - ENHANCED VERSION"""
        
        # Build command with explicit Python path
        cmd_parts = [python_path, "train_single.py"]
        
        # Add all arguments dynamically
        for key, value in cmd_args.items():
            # ✅ 替换成这部分
            if isinstance(value, bool):
                # 🔧 FIX: Always pass boolean values as strings
                cmd_parts.extend([f"--{key}", str(value).lower()])
            else:
                # Regular key-value pairs
                cmd_parts.extend([f"--{key}", str(value)])
        
        return " ".join(cmd_parts)
        
    def launch_experiments(self, experiments, base_args, device_assignment="auto", launch_delay=2):
        """Launch multiple experiments"""
        
        if not self.check_tmux_available():
            raise RuntimeError("tmux is not available. Please install tmux first.")
        
        print(f"📋 Launching {len(experiments)} experiments...")
        print(f"📁 Logs will be saved to: {self.log_dir}")
        print(f"⏰ Launch delay: {launch_delay}s between experiments")
        
        # Device assignment logic
        if device_assignment == "auto":
            try:
                import torch
                n_gpus = torch.cuda.device_count()
                if n_gpus > 0:
                    device_ids = list(range(n_gpus))
                    print(f"🖥️  Auto-detected {n_gpus} GPUs: {device_ids}")
                else:
                    device_ids = [-1]  # CPU
                    print("🖥️  No GPUs detected, using CPU")
            except ImportError:
                device_ids = [-1]
                print("🖥️  PyTorch not available, using CPU")
        elif isinstance(device_assignment, list):
            device_ids = device_assignment
        else:
            device_ids = [device_assignment]
        
        # Launch experiments
        launched = 0
        failed = 0
        
        for i, exp_config in enumerate(experiments):
            device_id = device_ids[i % len(device_ids)] if len(device_ids) > 0 else None
            
            success = self.launch_experiment(
                exp_config=exp_config,
                base_args=base_args,
                device_id=device_id,
                delay=launch_delay if i > 0 else 0
            )
            
            if success:
                launched += 1
            else:
                failed += 1
        
        print(f"\n📊 Launch Summary:")
        print(f"   ✅ Successfully launched: {launched}")
        print(f"   ❌ Failed to launch: {failed}")
        
        if launched > 0:
            print(f"\n📱 Monitoring Commands:")
            print(f"   List sessions: tmux ls")
            print(f"   Attach to session: tmux attach -t <exp_name>")
            print(f"   Kill session: tmux kill-session -t <exp_name>")
            print(f"   Kill all: tmux kill-server")
            print(f"   View logs: tail -f {self.log_dir}/stdout/<exp_name>.log")

    def get_session_status(self):
        """Get status of all running tmux sessions"""
        try:
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name},#{session_created},#{?session_attached,attached,detached}"],
                capture_output=True, text=True, check=True
            )
            
            sessions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        sessions.append({
                            'name': parts[0],
                            'created': parts[1],
                            'status': parts[2]
                        })
            return sessions
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
    
    def monitor_experiments(self, experiments):
        """Monitor running experiments"""
        exp_names = [exp["name"] for exp in experiments]
        
        while True:
            sessions = self.get_session_status()
            running_experiments = [s for s in sessions if s['name'] in exp_names]
            
            if not running_experiments:
                print("🏁 All experiments completed!")
                break
            
            print(f"\n📊 Running experiments ({len(running_experiments)}):")
            for session in running_experiments:
                print(f"   🔄 {session['name']} ({session['status']})")
            
            print(f"\n⏰ Checking again in 30 seconds... (Ctrl+C to stop monitoring)")
            try:
                time.sleep(30)
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped.")
                break


def define_predefined_experiments():
    """Define specific, hand-picked experiments - FIXED PARAMETERS"""
    experiments = [
        # Might be the best
        {
            "name": "Msk_CLS_AE",
            "params": {
                "pooling_type": "cls",
                "mlp_type": "attention_enhanced",
                "use_physics_mask": True,  # Use physics mask for this experiment  # Default threshold
                "use_cls_token": True,  # Use CLS token in the transformer
                "batch_size": 512,
                "num_workers":4  # Default batch size
            }
        },
        {
            "name": "Msk_CLS_moe",
            "params": {
                "pooling_type": "cls",
                "mlp_type": "mixture_of_experts",
                "use_physics_mask": True, 
                "mask_threshold": 2,  # Default threshold
                "use_cls_token": True,  # Use CLS token in the transformer
                "d_model": 64,
                "batch_size": 512,
                "num_workers": 2  # Default model dimension
            }
        }]
    
    return experiments


def generate_grid_search_experiments(param_grid):
    """Generate experiments from parameter grid"""
    keys, values = zip(*param_grid.items())
    experiments = []
    
    for i, v in enumerate(itertools.product(*values)):
        params = dict(zip(keys, v))
        
        # Create a readable name
        name_parts = []
        name_parts.append(f"Q{params['num_qubits']}")
        name_parts.append(f"D{params['d_model']}")
        name_parts.append(f"H{params.get('nhead', 8)}")
        name_parts.append(params['pooling_type'].upper())
        name_parts.append(params['mlp_type'].upper())
        if params.get('use_physics_mask', False):
            name_parts.append("PHYS")
        
        name = "_".join(name_parts)
        experiments.append({"name": name, "params": params})
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Launch distributed quantum magic experiments")
    
    # Experiment selection
    parser.add_argument("--experiments", type=str, nargs="+", 
                        help="Specific experiments to run (by name). If not specified, runs predefined list.")
    parser.add_argument("--list_experiments", action="store_true", 
                        help="List all available experiments and exit")
    parser.add_argument("--grid_search", action="store_true", 
                        help="Use grid search instead of predefined experiments")
    
    # Base configuration for all experiments
    parser.add_argument("--num_qubits", type=int, default=2, help="Number of qubits")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")   
    
    
    # Data configuration - IMPORTANT: Update these paths!
    parser.add_argument("--data_folder", default="/home/gwl/DURF/Transformer/dataset/Decoded_Tokens", type=str, help="Data folder path")
    parser.add_argument("--labels_path", default="/home/gwl/DURF/Transformer/dataset/Label/magic_labels_for_input_for_2_qubits_mixed_1_200000_datapoints.npy", type=str, help="Labels file path")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--num_workers", type=int, default=2, help="Data loading workers")
    
    # Launcher configuration
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--device_assignment", type=str, default="auto", 
                        help="Device assignment strategy: 'auto', 'cpu', or comma-separated GPU IDs")
    parser.add_argument("--launch_delay", type=int, default=2, 
                        help="Delay between launching experiments (seconds)")
    parser.add_argument("--dry_run", action="store_true", help="Show commands without executing")
    parser.add_argument("--monitor", action="store_true", help="Monitor experiments after launching")
    parser.add_argument("--kill_existing", action="store_true", help="Kill existing sessions before launching")
    
    args = parser.parse_args()
    
    # Define parameter grid for grid search
    param_grid = {
        'num_qubits': [2, 3], 
        'd_model': [32, 64], 
        'nhead': [4, 8],
        'pooling_type': ['cls', 'mean', 'attention', 'structured'],
        'mlp_type': ['standard', 'physics_aware'], 
        'use_physics_mask': [True, False],
        'mask_threshold': [1, 2],
    }
    
    # Get experiments based on user choice
    if args.grid_search:
        print("🔍 Generating experiments from grid search...")
        all_experiments = generate_grid_search_experiments(param_grid)
        print(f"📋 Generated {len(all_experiments)} experiments from grid search")
    else:
        all_experiments = define_predefined_experiments()
        print(f"📋 Using {len(all_experiments)} predefined experiments")
    
    # List experiments if requested
    if args.list_experiments:
        print("📋 Available experiments:")
        for i, exp in enumerate(all_experiments, 1):
            print(f"   {i:2d}. {exp['name']}")
            if 'params' in exp:
                for key, value in exp['params'].items():
                    print(f"       {key}: {value}")
            print()
        return
    
    # Filter experiments if specific ones requested
    if args.experiments:
        experiment_names = set(args.experiments)
        experiments = [exp for exp in all_experiments if exp['name'] in experiment_names]
        
        # Check for invalid experiment names
        found_names = set(exp['name'] for exp in experiments)
        invalid_names = experiment_names - found_names
        if invalid_names:
            print(f"❌ Invalid experiment names: {invalid_names}")
            print("Use --list_experiments to see available experiments")
            return
    else:
        experiments = all_experiments
    
    # Parse device assignment
    if args.device_assignment == "auto":
        device_assignment = "auto"
    elif args.device_assignment == "cpu":
        device_assignment = [-1]
    else:
        try:
            device_assignment = [int(x.strip()) for x in args.device_assignment.split(',')]
        except ValueError:
            print(f"❌ Invalid device assignment: {args.device_assignment}")
            return
    
    # Prepare base arguments for all experiments
    base_args = {
        "num_qubits": args.num_qubits,
        "d_model": args.d_model,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "data_folder": args.data_folder,
        "labels_path": args.labels_path,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_workers": args.num_workers,
    }
    
    # Create dispatcher
    dispatcher = ExperimentDispatcher(
        log_dir=args.log_dir,
        dry_run=args.dry_run
    )
    
    # Kill existing sessions if requested
    if args.kill_existing:
        print("🔪 Killing existing tmux sessions...")
        for exp in experiments:
            dispatcher.kill_session(exp["name"])
    
    # Launch experiments
    dispatcher.launch_experiments(
        experiments=experiments,
        base_args=base_args,
        device_assignment=device_assignment,
        launch_delay=args.launch_delay
    )
    
    # Monitor if requested
    if args.monitor and not args.dry_run:
        print("\n🔍 Starting experiment monitoring...")
        dispatcher.monitor_experiments(experiments)


if __name__ == "__main__":
    main()