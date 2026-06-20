import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from main import main

sys.exit(main())
