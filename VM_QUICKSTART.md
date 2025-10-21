# VM Quick Start

After your container restarts, use one of these methods:

## Method 1: One-Line (Fastest) ⚡

```bash
source /workspace/ball-action-spotting/quick_setup.sh
```

That's it! Environment activated and ready to use.

---

## Method 2: Auto-Setup Script (Recommended) 🔧

```bash
bash /workspace/ball-action-spotting/vm_setup.sh
```

This will:
- ✅ Check if environment exists (create if missing)
- ✅ Activate conda environment
- ✅ Verify all imports work
- ✅ Check GPU status
- ✅ Show data/model status
- ✅ Display usage examples

---

## Method 3: Manual (If Scripts Fail) 🛠️

```bash
cd /workspace/ball-action-spotting
conda activate ball-action-spotting
python mvp/run_mvp.py --help
```

---

## Make It Automatic

Add to your `~/.bashrc` to auto-activate on login:

```bash
echo 'source /workspace/ball-action-spotting/quick_setup.sh' >> ~/.bashrc
```

Now every time you log in, the environment will be ready! 🎯

---

## Full Documentation

See [VM_RESTART_SETUP.md](VM_RESTART_SETUP.md) for complete troubleshooting guide.
