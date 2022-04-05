import nextcord
from nextcord.ext import commands
import os
import cv2
from tool import darknet2pytorch
import torch
from tool.utils import plot_boxes_cv2
from tool.torch_utils import do_detect

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
async def detect(ctx):
    await ctx.send(ctx.author.id)

with open("token.txt", "r") as f:
    token = f.read()

bot.run(token)