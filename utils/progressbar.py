import math

class ProgressBar:
    def __init__(self, maxStep=150, fill="#"):
        self.maxStep = maxStep
        self.fill = fill
        self.barLength = 20
        self.barInterval = 0
        self.prInterval = 0
        self.count = 0
        self.progress = 0
        self.barlenSmaller = True
        self.genBarCfg()

    def genBarCfg(self):
        if self.maxStep >= self.barLength:
            self.barInterval = math.ceil(self.maxStep / self.barLength)
        else:
            self.barlenSmaller = False
            self.barInterval = math.floor(self.barLength / self.maxStep)
        self.prInterval = 100 / self.maxStep

    def resetBar(self):
        self.count = 0
        self.progress = 0

    def updateBar(self, step, headData={'head':10}, endData={'end_1':2.2, 'end_2':1.0}, keep=False):
        head_str = "\r"
        end_str = " "
        process = ""
        if self.barlenSmaller:
            if step != 0 and step % self.barInterval == 0:
                self.count += 1
        else:
            self.count += self.barInterval
        self.progress += self.prInterval
        for key in headData.keys():
            head_str = head_str + key + ": " + str(headData[key]) + " "
        for key in endData.keys():
            end_str = end_str + key + ": " + str(endData[key]) + " "
        if step == self.maxStep:
            process += head_str
            process += "[%3s%%]: [%-20s]" % (100.0, self.fill * self.barLength)
            process += end_str
            if not keep:
                process += "\n"
        else:
            process += head_str
            process += "[%3s%%]: [%-20s]" % (round(self.progress, 1), self.fill * self.count)
            process += end_str
        print(process, end='', flush=True)


def train_update_pbar(bar: ProgressBar, e, i, train_loss, train_mae):
    if i != bar.maxStep-1:
        bar.updateBar(
                i+1, headData={'Epoch':e+1, 'Status':'training'}, 
                endData={
                    'Train loss': "{:.5f}".format(train_loss/(i+1)),
                    'Train MAE': "{:.5f}".format(train_mae/(i+1))})
    else:
        bar.updateBar(
                i+1, headData={'Epoch':e+1, 'Status':'finished'}, 
                endData={
                    'Train loss': "{:.5f}".format(train_loss/(i+1)),
                    'Train MAE': "{:.5f}".format(train_mae/(i+1))})

def test_update_pbar(bar: ProgressBar, e, i, test_loss, test_mae):
    if i != bar.maxStep-1:
        bar.updateBar(
                i+1, headData={'Epoch (Test)':e+1, 'Status':'testing'}, 
                endData={
                    'Test loss': "{:.5f}".format(test_loss/(i+1)),
                    'Test MAE': "{:.5f}".format(test_mae/(i+1))})
    else:
        bar.updateBar(
                i+1, headData={'Epoch (Test)':e+1, 'Status':'finished'}, 
                endData={
                    'Test loss': "{:.5f}".format(test_loss/(i+1)),
                    'Test MAE': "{:.5f}".format(test_mae/(i+1))})