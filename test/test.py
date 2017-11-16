from PIL import Image

bg_img_file = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX/raw/C0001273.jpg'
fg_img_file = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX/mask/C0001273_EX.png'

bg_img = Image.open(bg_img_file)
fg_img = Image.open(fg_img_file)
fg_img_rgb = Image.new('RGB', fg_img.size)
fg_img_rgb.paste(fg_img)

bg_img = bg_img.resize((1024,1024))
fg_img_rgb = fg_img_rgb.resize((1024,1024))

Image.blend(bg_img, fg_img_rgb, 0.5).show()


from pynput import mouse

def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))

def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    if not pressed:
        # Stop listener
        return False

def on_scroll(x, y, dx, dy):
    print('Scrolled {0} at {1}'.format(
        'down' if dy < 0 else 'up',
        (x, y)))

# Collect events until released
with mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll) as listener:
    listener.join()