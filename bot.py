from email import message
from pydoc import classname
import nextcord
from nextcord.ext import commands
import os
import cv2
from tool import darknet2pytorch
import torch
from tool.utils import plot_boxes_cv2
from tool.torch_utils import do_detect
from urllib.request import urlopen, Request
import numpy as np

model_pt = darknet2pytorch.Darknet('yolov4-obj.cfg', inference=True)
model_pt.load_state_dict(torch.load('yolov4-pytorch.pth'))

bot = commands.Bot(command_prefix='/')
bot.remove_command("help")

@bot.event
async def on_ready():
    print("bot is now ready")

@bot.command()
async def ping(ctx):
    await ctx.send("pong")

@bot.command() 
# make a detect commands that grab the image from the url and then do the detect
async def detect(ctx):
    # get the image from the message
    print(ctx.message.attachments[0].url)
    img_url = ctx.message.attachments[0].url
    processing_embed = embed = nextcord.Embed(title="Processing...", color=0x00ff00)
    await ctx.send(embed=processing_embed)
    # delete the message processing embed
    await ctx.message.delete()
    # get the image from the url
    req = Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
    img_data = urlopen(req).read()
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    # do the detect
    boxes = do_detect(model_pt, cv2.resize(img, (416, 416)), 0.5, 0.4, use_cuda=False)
    # plot the boxes
    plot_boxes_cv2(img, boxes[0], "detect.jpg", class_names=["furry"])
    # send the image in a cool embed
    embed = nextcord.Embed(title="Results", description=f"furries detected: {len(boxes[0])}", color=0x00ff00)
    embed.set_image(url="attachment://detect.jpg")
    await ctx.send(embed=embed, file=nextcord.File("detect.jpg", "detect.jpg"))
    # delete the image
    os.remove("detect.jpg")

# add a help command
@bot.command()
async def help(ctx):
    embed = nextcord.Embed(title="Help", description="This is a help command", color=0x00ff00)
    embed.add_field(name="ping", value="pong", inline=False)
    embed.add_field(name="detect", value="detect", inline=False)
    await ctx.send(embed=embed)

with open("token.txt", "r") as f:
    token = f.read()

bot.run(token)