import matplotlib.pyplot as plt


class PointSelector:
    def __init__(self, image_array):
        self.image_array = image_array
        self.points = []  # To store the selected points

    def onclick(self, event):
        if len(self.points) < 3:
            x, y = event.xdata, event.ydata
            self.points.append((x, y))
            plt.plot(x, y, 'ro')  # Plot the point as a red dot
            plt.draw()  # Update the plot with the new point
            print(f"Point selected at: ({x:.2f}, {y:.2f})")
            if len(self.points) == 3:
                plt.pause(0.3)
                plt.close()  # Close the plot after selecting 2 points

    def select_points(self):
        # Display the image
        plt.imshow(self.image_array, cmap='gray')
        plt.title('Select head origin and then head dest, then tail tip')

        # Connect the click event to the handler
        cid = plt.gcf().canvas.mpl_connect('button_press_event', self.onclick)

        # Show the plot
        plt.show()

        # Disconnect the event
        plt.gcf().canvas.mpl_disconnect(cid)

        return self.points


