import matplotlib.pyplot as plt

def plot_history(history, output_plot):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(history["train_loss"], label="train_loss")
	plt.plot(history["val_loss"], label="val_loss")
	plt.plot(history["train_acc"], label="train_acc")
	plt.plot(history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(output_plot)