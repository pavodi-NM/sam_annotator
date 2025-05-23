"""Main entry point for SAM Annotator."""

# Just import and re-export the main function from the cli package
from sam_annotator.cli.main import main
import sys

# Pass sys.argv to main when called as entry point
if __name__ == "__main__": 
    main(sys.argv[1:])
else:
    # This is called when used as a console script entry point
    def cli_entry_point():
        """Entry point for console script."""
        main(sys.argv[1:])