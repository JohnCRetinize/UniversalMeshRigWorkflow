import os
import platform
import urllib.request
import zipfile
import shutil
import subprocess
from pathlib import Path

class AutoRigInstaller:
    """Manages the portable Blender environment."""
    
    VERSION = "4.2.0"
    
    @classmethod
    def get_paths(cls, base_path):
        bin_dir = Path(base_path) / "bin"
        blender_dir = bin_dir / f"blender-{cls.VERSION}"
        
        if platform.system() == "Windows":
            exe = blender_dir / "blender.exe"
            python = blender_dir / "4.2" / "python" / "bin" / "python.exe"
        else:
            exe = blender_dir / "blender"
            python = blender_dir / "4.2" / "python" / "bin" / "python3.11"
            
        return bin_dir, blender_dir, exe, python

    @classmethod
    def ensure_blender(cls, base_path):
        bin_dir, blender_dir, exe, python = cls.get_paths(base_path)
        
        if exe.exists():
            return exe, python

        print(f"### [AutoRig] Installing Portable Blender {cls.VERSION}...")
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        url = f"https://mirrors.dotsrc.org/blender/release/Blender4.2/blender-{cls.VERSION}-windows-x64.zip"
        if platform.system() != "Windows":
            # For brevity, assuming Windows, but URL can be adjusted for Linux
            url = f"https://mirrors.dotsrc.org/blender/release/Blender4.2/blender-{cls.VERSION}-linux-x64.tar.xz"

        zip_p = bin_dir / "blender_pkg.zip"
        urllib.request.urlretrieve(url, zip_p)
        
        with zipfile.ZipFile(zip_p, 'r') as z:
            z.extractall(bin_dir)
            
        # Cleanup folder names
        for d in bin_dir.glob("blender-*"):
            if d.is_dir() and d != blender_dir:
                d.rename(blender_dir)
        
        zip_p.unlink()
        
        print(f"### [AutoRig] Injecting Math Dependencies...")
        subprocess.check_call([str(python), "-m", "pip", "install", "scipy", "scikit-learn", "numpy"])
        
        return exe, python