import argparse



def parse_args(args=None, namespace=None):
	parser = argparse.ArgumentParser(
		description='Train and Validate DSBN Network.\n' + \
                    'target label:0, sorce label:1,2,... \n' + \
                    '[digits: svhn, mnist, usps || ' + \
                    'office: amazon, webcam, dslr || ' + \
                    'office-home: Art, Clipart, Product, RealWorld || ' + \
                    'imageCLEF: caltech, pascal, imagenet || ' + \
                    'visDA: train, validation]')

	parser.add_argument('--model-name',
                        help="model name ['lenet',  'resnet50', 'resnet50dsbn', 'resnet101', 'resnet101dsbn']",
                        default='resnet50', type=str)
	parser.add_argument('--exp-setting', help='exp setting[digits, office, imageclef, visda]', default='office',
                        type=str)
	parser.add_argument('--init-model-path', help='init model path', default='', type=str)
	parser.add_argument('--save-dir', help='directory to save models', default='output/office_default', type=str)

	args = parser.parse_args(args=args, namespace=namespace)
	return args

args = parse_args()
print(args)
print(args.model_name)
print(args.exp_setting)
print(args.save_dir)
