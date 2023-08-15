
def plot_bboxes(img,
    bboxes: List[List[float]],
    damage: List[str],
    labels: Optional[List[str]] = None) -> None:

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        npimg = np.array(img)
        plt.imshow(npimg, interpolation='nearest')

        for i, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin

            if damage[i] == 1:
                box = patches.Rectangle((xmin, ymin), w, h, edgecolor="yellow", facecolor="none")
            else:
                box = patches.Rectangle((xmin, ymin), w, h, edgecolor="red", facecolor="none")

            ax.add_patch(box)

            if labels is not None:
                      rx, ry = box.get_xy()
                      cx = rx + box.get_width()/10.0
                      cy = ry + box.get_height()/10.0
                      l = ax.annotate(
                        labels[i],
                        (cx, cy),
                        fontsize=6,
                        fontweight="bold",
                        color="black",
                        ha='center',
                        va='center'
                      )

            if damage[i] == 1:
                l.set_bbox(dict(facecolor='yellow', alpha=0.4, edgecolor='yellow'))
            else:
                l.set_bbox(dict(facecolor='red', alpha=0.4, edgecolor='red'))

        plt.axis('off')
        return fig
