from glasses_detector import GlassesClassifier, GlassesDetector

# Prints either '1' or '0'
classifier = GlassesClassifier()
classifier.process_file(
    input_path=r"D:/Projects/Eye Glasses Detection/github star/specs.jpg",     # can be a list of paths
    format={True: "1", False: "0"},   # similar to format="int"
    show=True,                        # to print the prediction
)

# Opens a plot in a new window
detector = GlassesDetector()
detector.process_file(
    input_path=r"D:/Projects/Eye Glasses Detection/github star/specs.jpg",          # can be a list of paths
    format="img",                     # to return the image with drawn bboxes
    show=True,                        # to show the image using matplotlib
)