import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-o", "--ouput", type=str,
	help="path to output")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum confidence")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
ap.add_argument("--html", 
    help="export to html")
ap.add_argument("-m", "--method", type=int, default=1,
	help="method")
args = vars(ap.parse_args())