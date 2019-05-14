import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default='../stock/don-thuoc.png',
	help="path to input image")
ap.add_argument("-o", "--output", type=str, default=None,
	help="path to output")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum confidence")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
ap.add_argument("--html",  action='store_true',
    help="export to html")
ap.add_argument("--noboder",  action='store_true',
    help="remove html bolder")
ap.add_argument("-m", "--method", type=int, default=1,
	help="method")
ap.add_argument("-ot", "--output-type", type=str, default="xml",
	help="output-type xml or string")

args = vars(ap.parse_args())

#Import Method
if args["method"] == 0:
	from converter.med0.converter import Converter
elif args["method"] == 1:
	from converter.med1.converter import Converter
elif  args["method"] == 2:
	from converter.med2.converter import Converter

con = Converter(file = args['image'], lang = 'vie', output_type=args['output_type'])
output = con.execute()



#Write to file
if args['output']:
	text_file = open(args['output'], "w")
	text_file.write(output)
	text_file.close()

	#Execute HTML
	if args['html']:
		f = open(args['output'], "r")
		temp = f.read().replace('</body>','<script src="https://unpkg.com/hocrjs"></script></body>')
		f.close()
		f = open(args['output'], "w")
		f.write(temp)
		f.close()

	#Add unbolder CSS
	if args['noboder']:
            f = open(args['output'], "r")
            rpstr = """
            <head>
            <style>
            * {border-style: none !important;}
            </style>
            
            """
            temp = f.read().replace('<head>',rpstr)
            f.close()
            f = open(args['output'], "w")
            f.write(temp)
            f.close()
else:
	print(output)
