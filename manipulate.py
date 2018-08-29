from PIL import Image, ImageDraw, ImageOps
import time
import sys
import colorsys
from math import sqrt
from operator import itemgetter


# Find distance between two values
def distance(x, y):
    return abs(x-y)


# Performs addition between two images
def add(image1, image2):

    # Loads input images
    image1_pixels = image1.load()
    image2_pixels = image2.load()

    # Creates output
    output = Image.new("RGBA", image1.size)
    draw = ImageDraw.Draw(output)

    # Adds each channel value in pixel by corresponding value in second image
    for x in range(image1.width):
        for y in range(image1.height):
            r_1, g_1, b_1, a_1 = image1_pixels[x, y]
            for m in range(image2.width):
                for n in range(image2.height):
                    r_2, g_2, b_2, a_2 = image2_pixels[m, n]
                    if x == m and y == n:
                        r = int(r_1 + r_2)
                        g = int(g_1 + g_2)
                        b = int(b_1 + b_2)
                        a = int(255)
                        draw.point((m, n), (r, g, b, a))

    return output


# Performs invert
def invert(image):

    # Opens, loads and displays image
    input_pixels = image.load()

    # Creates output
    output = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(output)

    # Creates image with gamma correction
    for x in range(output.width):
        for y in range(output.height):
            r, g, b, a = input_pixels[x, y]
            r = int(255 - r)
            g = int(255 - g)
            b = int(255 - b)
            a = int(255 - a)
            draw.point((x, y), (r, g, b, a))

    return output


# Performs multiplication between an image and a constant
def multiply1(image, value):

    # Loads input image
    pixels = image.load()

    # Creates output
    output = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(output)

    # Multiplies each channel value in pixel by a constant
    for x in range(image.width):
        for y in range(image.height):
            r, g, b, a = pixels[x, y]
            r = int(r * value)
            g = int(g * value)
            b = int(b * value)
            a = int(255)
            draw.point((x, y), (r, g, b, a))

    return output


# Performs multiplication between two images (uses alpha of foreground (image1) image)
def multiply2(image1, image2):

    # Loads input images
    image1_pixels = image1.load()
    image2_pixels = image2.load()

    # Creates output
    output = Image.new("RGBA", image1.size)
    draw = ImageDraw.Draw(output)

    # Adds each channel value in pixel by corresponding value in second image
    for x in range(image1.width):
        for y in range(image1.height):
            r_1, g_1, b_1, a_1 = image1_pixels[x, y]
            for m in range(image2.width):
                for n in range(image2.height):
                    r_2, g_2, b_2, a_2 = image2_pixels[m, n]
                    if x == m and y == n:
                        r = int(((r_1 / 255) * (r_2 / 255)) * 255)
                        g = int(((g_1 / 255) * (g_2 / 255)) * 255)
                        b = int(((b_1 / 255) * (b_2 / 255)) * 255)
                        a = int(a_1)
                        draw.point((m, n), (r, g, b, a))

    return output


# Performs gamma correction
def gamma():

    # Opens, loads and displays image
    image = Image.open("girl.jpg")
    pixels = image.load()
    image.show()

    # Creates output
    output = Image.new("RGB", image.size)
    draw = ImageDraw.Draw(output)

    # Gets gamma value
    gamma_value = input("Enter a gamma value between 0.01 and 7.99: ")

    # Calculates gamma correction
    gamma_correction = 1 / float(gamma_value)

    # Creates image with gamma correction
    for x in range(output.width):
        for y in range(output.height):
            r, g, b = pixels[x, y]
            # Assigns new value to R, G, B
            r = int(255 * (r / 255) ** gamma_correction)
            g = int(255 * (g / 255) ** gamma_correction)
            b = int(255 * (b / 255) ** gamma_correction)
            draw.point((x, y), (r, g, b))

    # Shows manipulated image
    output.show()

    print("Performed Gamma Correction.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs contrast
def contrast():

    # Opens, loads and displays image
    image = Image.open("girl.jpg")
    input_pixels = image.load()
    image.show()

    # Creates output
    output = Image.new("RGB", image.size)
    draw = ImageDraw.Draw(output)

    # Finds minimum/maximum color values
    imin = 255
    imax = 0
    for x in range(image.width):
        for y in range(image.height):
            r, g, b = input_pixels[x, y]
            i = (r + g + b) / 3
            imin = min(imin, i)
            imax = max(imax, i)

    # Creates image with increased contrast
    for x in range(output.width):
        for y in range(output.height):
            r, g, b = input_pixels[x, y]
            # Current luminosity
            i = (r + g + b) / 3
            # New luminosity
            ip = 255 * (i - imin) / (imax - imin)
            r = int(r * ip / i)
            g = int(g * ip / i)
            b = int(b * ip / i)
            draw.point((x, y), (r, g, b))

    # Shows manipulated image
    output.show()

    print("Performed Contrast.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs sobel operation (edge-detection)
def sobel():

    # Opens, loads and displays image
    image = Image.open("girl.jpg")
    pixels = image.load()
    image.show()

    # Create output
    output = Image.new("RGB", image.size)
    draw = ImageDraw.Draw(output)

    # Calculates pixel intensity
    intensity = [[sum(pixels[x, y]) / 3 for y in range(image.height)] for x in range(image.width)]

    # Sobel kernels
    kernelx = [[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]]
    kernely = [[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]

    # Computes convolution
    for x in range(1, image.width - 1):
        for y in range(1, image.height - 1):
            magx, magy = 0, 0
            for a in range(3):
                for b in range(3):
                    xn = x + a - 1
                    yn = y + b - 1
                    magx += intensity[xn][yn] * kernelx[a][b]
                    magy += intensity[xn][yn] * kernely[a][b]

            # Draw in black and white the magnitude
            color = int(sqrt(magx ** 2 + magy ** 2))
            draw.point((x, y), (color, color, color))

    # Shows manipulated image
    output.show()

    print("Performed Sobel Operation.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs traditional edge detection
def edgeDetect():

    # Opens, loads and displays image
    image = Image.open("girl.jpg")
    pixels = image.load()
    image.show()

    # Creates output
    output = Image.new("RGB", image.size)
    draw = ImageDraw.Draw(output)

    # Edge detection kernel
    kernel = [[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1]]

    # Computes convolution
    for x in range(1, image.width - 1):
        for y in range(1, image.height - 1):
            acc = [0, 0, 0]
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = x + a - 1
                    yn = y + b - 1
                    pixel = pixels[xn, yn]
                    acc[0] += pixel[0] * kernel[a][b]
                    acc[1] += pixel[1] * kernel[a][b]
                    acc[2] += pixel[2] * kernel[a][b]

            draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))

    # Shows manipulated image
    output.show()

    print("Performed Edge Detection.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs blur
def blur():

    # Opens, loads and displays image
    image = Image.open("girl.jpg")
    pixels = image.load()
    image.show()

    # Creates output
    output = Image.new("RGB", image.size)
    draw = ImageDraw.Draw(output)

    # Box Blur kernel
    box_kernel = [[1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9]]

    # Gaussian kernel
    gaussian_kernel = [[1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256],
                       [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                       [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
                       [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                       [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256]]

    # Choose blur kernel
    chosen_kernel = input("Type 'b' to perform a box blur or type 'g' to perform a gaussian blur: ")

    # Select kernel here
    if chosen_kernel == "b":
        kernel = box_kernel
    elif chosen_kernel == "g":
        kernel = gaussian_kernel
    else:
        print("Error: You did not type either 'b' or 'g'.")
        sys.exit()

    # Middle of kernel
    offset = len(kernel) // 2

    # Computes convolution
    for x in range(offset, image.width - offset):
        for y in range(offset, image.height - offset):
            acc = [0, 0, 0]
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = x + a - offset
                    yn = y + b - offset
                    pixel = pixels[xn, yn]
                    acc[0] += pixel[0] * kernel[a][b]
                    acc[1] += pixel[1] * kernel[a][b]
                    acc[2] += pixel[2] * kernel[a][b]

            draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))

    # Shows manipulated image
    output.show()

    print("Performed Blur.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs sharpen
def sharpen():

    # Opens, loads and displays image
    image = Image.open("girl.jpg")
    pixels = image.load()
    image.show()

    # Creates output
    output = Image.new("RGB", image.size)
    draw = ImageDraw.Draw(output)

    # Sharpening kernel
    kernel = [[-1, -1, -1],
              [-1, 9, -1],
              [-1, -1, -1]]

    # Computes convolution
    for x in range(1, image.width - 1):
        for y in range(1, image.height - 1):
            acc = [0, 0, 0]
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = x + a - 1
                    yn = y + b - 1
                    pixel = pixels[xn, yn]
                    acc[0] += pixel[0] * kernel[a][b]
                    acc[1] += pixel[1] * kernel[a][b]
                    acc[2] += pixel[2] * kernel[a][b]

            draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))

    # Shows manipulated image
    output.show()

    print("Performed Sharpen.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs median spatial filter
def medSpatialFilter():

    # Opens, loads and displays image
    image = Image.open("girl.jpg")
    pixels = image.load()
    image.show()

    # Creates output
    output = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(output)

    lum_list = []

    for x in range(1, image.width - 1):
        for y in range(1, image.height - 1):
            kernel = [[pixels[x - 1, y - 1], pixels[x, y - 1], pixels[x + 1, y - 1]],
                      [pixels[x - 1, y], pixels[x, y], pixels[x + 1, y]],
                      [pixels[x - 1, y + 1], pixels[x, y + 1], pixels[x + 1, y + 1]]]
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = x + a - 1
                    yn = y + b - 1
                    r, g, b = pixels[xn, yn]
                    # Finds luminance value
                    luminance = (r * 0.2126 + g * 0.7152 + b * 0.0722)
                    tup = (a, b, luminance)
                    lum_list.append(tup)
                sorted(lum_list, key=itemgetter(2))
                sorted_position_a = [tup[0] for tup in lum_list]
                sorted_position_b = [tup[1] for tup in lum_list]
                r, g, b = kernel[sorted_position_a[4]][sorted_position_b[4]]
                draw.point((x, y), (r, g, b))

    # Shows manipulated image
    output.show()

    print("Performed Median Spatial Filter.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs mix
def mix():

    # Loads and displays images
    foreground = Image.open("bird.jpg")
    foreground = foreground.convert("RGBA")

    background = Image.open("checkerboard.png")
    background = background.convert("RGBA")

    # Calculates MV and (1-MV)
    percent = input("What percentage would you like the foreground to show through?  ")
    fg_mv = int(percent) / 100
    bg_mv = 1 - fg_mv

    # Computes O = (MV * A) + [(1 - MV) * B]
    fg_multiplied = multiply1(foreground, fg_mv)
    bg_multiplied = multiply1(background, bg_mv)
    output = add(fg_multiplied, bg_multiplied)

    # Shows manipulated image
    output.show()

    print("Performed Mix.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs key-mix
def keyMix():

    # Loads and displays images
    foreground = Image.open("bird.jpg")
    foreground = foreground.convert("RGBA")

    background = Image.open("checkerboard.png")
    background = background.convert("RGBA")

    foreground_matte = Image.open("bird_matte.jpg")
    foreground_matte = foreground_matte.convert("RGBA")

    # Computes O = (A * M) + [(1 - M) * B]
    new_foreground = multiply2(foreground_matte, foreground)
    inverted_matte = invert(foreground_matte)
    new_background = multiply2(inverted_matte, background)
    output = add(new_foreground, new_background)

    # Shows manipulated image
    output.show()

    print("Performed Key-Mix.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs over operation
def over():

    # Loads and displays images
    foreground = Image.open("bird.jpg")
    foreground = foreground.convert("RGBA")

    background = Image.open("checkerboard.png")
    background = background.convert("RGBA")

    foreground_matte = Image.open("bird_matte.jpg")
    foreground_matte = foreground_matte.convert("RGBA")

    # Computes O = A + [(1 - A) * B]
    premult_foreground = multiply2(foreground_matte, foreground)
    inverted_matte = invert(foreground_matte)
    new_bg = multiply2(inverted_matte, background)
    output = add(premult_foreground, new_bg)

    # Shows manipulated image
    output.show()

    print("Performed Over Operator.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs screen
def screen():

    # Loads and displays images
    foreground = Image.open("bird.jpg")
    foreground = foreground.convert("RGBA")

    background = Image.open("checkerboard.png")
    background = background.convert("RGBA")

    # Computes O = 1 - [(1 - A) * (1 - B)]
    inverted_foreground = invert(foreground)
    inverted_background = invert(background)
    mult_fg_bg = multiply2(inverted_foreground, inverted_background)
    output = invert(mult_fg_bg)

    # Shows manipulated image
    output.show()

    print("Performed Screen Operator.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs luma-key
def lumaKey():

    # Loads and displays images
    foreground = Image.open("leaf.jpg")
    foreground = foreground.convert("RGBA")
    pixels = foreground.load()

    background = Image.open("checkerboard.png")
    background = background.convert("RGBA")

    # Creates alpha masked image
    matte = Image.new("RGBA", foreground.size)
    draw = ImageDraw.Draw(matte)

    for x in range(foreground.width):
        for y in range(foreground.height):
            r, g, b, a = pixels[x, y]
            # Finds luminance value
            luminance = (r * 0.2126 + g * 0.7152 + b * 0.0722)
            # Finds distance to white
            d = distance(luminance, 255)
            # Masking using luminance threshold
            if 225 < d < 255:
                r = 0
                g = 0
                b = 0
                a = 0
            else:
                r = 255
                g = 255
                b = 255
                a = 255
            draw.point((x, y), (r, g, b, a))

    new_foreground = multiply2(matte, foreground)
    inverted_matte = invert(matte)
    new_background = multiply2(inverted_matte, background)
    output = add(new_foreground, new_background)

    # Shows manipulated image
    output.show()

    print("Performed Luma-Keying.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs chroma-key
def chromaKey():

    # Loads and displays images
    foreground = Image.open("dhouse.png")
    foreground = foreground.convert("RGBA")
    pixels = foreground.load()

    background = Image.open("checkerboard.png")
    background = background.convert("RGBA")

    # Creates alpha masked image
    matte = Image.new("RGBA", foreground.size)
    draw = ImageDraw.Draw(matte)

    for x in range(foreground.width):
        for y in range(foreground.height):
            r, g, b, a = pixels[x, y]
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            h = h * 360
            s = s * 100
            v = v * 100
            # Alpha-masking using hue threshold (green)
            if 90 < h < 150:
                h = s = v = a = 0
            else:
                h = s = 0
                v = 100
                a = 255
            r, g, b = colorsys.hsv_to_rgb((h / 360), (s / 100), (v / 100))
            draw.point((x, y), (int(r * 255), int(g * 255), int(b * 255), a))

    new_foreground = multiply2(matte, foreground)
    inverted_matte = invert(matte)
    new_background = multiply2(inverted_matte, background)
    output = add(new_foreground, new_background)

    # Shows manipulated image
    output.show()

    print("Performed Chroma-Keying.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Performs color difference method
def colorDifference():

    # Loads and displays images
    foreground = Image.open("dhouse.png")
    foreground = foreground.convert("RGBA")
    pixels = foreground.load()

    background = Image.open("checkerboard.png")
    background = background.convert("RGBA")

    # Creates alpha masked image
    matte = Image.new("RGBA", foreground.size)
    draw = ImageDraw.Draw(matte)

    # Performs alpha masking
    for x in range(foreground.width):
        for y in range(foreground.height):
            r, g, b, a = pixels[x, y]
            # matte creation
            a = g - max(b, r)
            if a > 0:
                r = g = b = a = 255
            else:
                r = g = b = a = 0
            draw.point((x, y), (r, g, b, a))

    # Creates spill suppressed image
    ss_foreground = Image.new("RGBA", foreground.size)
    draw = ImageDraw.Draw(ss_foreground)

    # Performs spill suppression
    for x in range(foreground.width):
        for y in range(foreground.height):
            r, g, b, a = pixels[x, y]
            # spill suppression
            if g > b:
                g = b
            else:
                g = g
            draw.point((x, y), (r, g, b, a))

    inverted_matte = invert(matte)
    new_foreground = multiply2(inverted_matte, ss_foreground)
    new_background = multiply2(matte, background)
    output = add(new_foreground, new_background)

    # Shows manipulated image
    output.show()

    print("Performed Color Difference Method.")

    key = input("Press 'w' to write or 'q' to quit. ")
    handleKey(key, output)

    return


# Writes image
def write(image):

    time_string = time.strftime("%Y.%m.%d_at_%H.%M.%S")

    image.save("output_" + time_string + ".png")
    print("Image saved as...  output_" + time_string + ".png.")
    return


# Decides what to do after keyboard key is clicked
def handleKey(key, image):

    # input detection
    if key == "g":
        gamma()
        return
    if key == "c":
        contrast()
        return
    if key == "e":
        edgeDetect()
        return
    if key == "b":
        blur()
        return
    if key == "s":
        sharpen()
        return
    if key == "f":
        medSpatialFilter()
        return
    if key == "m":
        mix()
        return
    if key == "k":
        keyMix()
        return
    if key == "o":
        over()
        return
    if key == "n":
        screen()
        return
    if key == "l":
        lumaKey()
        return
    if key == "h":
        chromaKey()
        return
    if key == "d":
        colorDifference()
        return
    if key == "w":
        write(image)
        return
    if key == "q":
        print("\nProgram terminated.\n")
        sys.exit()
    else:
        print("\nError: You did not choose an recognized key\n.")
        main()
    return


# Main
def main():

    image = Image.open("girl.jpg")

    # print instructions
    print("\nWelcome to Mandy's Image Manipulator.\n")
    print("Press...")
    print(" 'g' to Gamma Correct            'c' to Contrast")
    print(" 'g' for Sobel Operator          'e' for Edge Detection")
    print(" 'b' to Blur                     's' to Sharpen")
    print(" 'f' for Median Spatial Filter   'm' to Mix")
    print(" 'k' to Key Mix                  'o' for the Over Operator")
    print(" 'n' for the Screen Operator     'l' for Luma Key")
    print(" 'h' for Chroma Key              'd' for Color Difference\n")
    print("  Click 'w' to write the image and 'q' to quit.\n\n")

    key = input("Input Key: ")
    handleKey(key, image)


main()
