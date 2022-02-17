import numpy


class Highlighter(object):
    def __init__(self):
        self.color_map = [
            "#ff0000",
            "#ff4000",
            "#ff8000",
            "#ffbf00",
            "#ffff00",
            "#bfff00",
            "#80ff00",
            "#40ff00",
            "#00ff00",
            "#00ff40",
            "#00ff80",
            "#00ffbf",
            "#00ffff",
            "#00bfff",
            "#0080ff",
            "#0040ff",
            "#0000ff",
            "#4000ff",
            "#8000ff",
            "#bf00ff",
            "#ff00ff",
            "#ff00bf",
            "#ff0080",
            "#ff0040",
            "#ff0000",
        ]

    def highlight(self, activation, text):
        activation = activation.detach().cpu().numpy()

        step_size = (max(activation) - min(activation)) / len(self.color_map)

        lookup = numpy.array(list(numpy.arange(min(activation), max(activation), step_size)))

        colors = []

        for i, act in enumerate(activation):

            try:
                colors.append(self.color_map[numpy.where(act > lookup)[0][-1]])
            except IndexError:
                colors.append(len(self.color_map) - 1)

        str_ = "<br><br>"

        for i, (char, color) in enumerate(zip(list(text), colors)):
            str_ += self._render(char, color)

            if i % 100 == 0 and i > 0:
                str_ += "<br>"

        return str_

    def highlight_selection(self, activations, text, file_="resources/data/highlight.html", n=10):

        ix = numpy.random.choice(activations.shape[1], size=n)

        rendered = ""

        for i in ix:

            rendered += self.highlight(activations[:, i], text)

        with open(file_, "w") as f:
            f.write(rendered)

    @staticmethod
    def _render(char, color):
        return '<span style="background-color: {}">{}</span>'.format(color, char)
