from graphviz import Digraph

def create_resnet34_unet_diagram():
    dot = Digraph(comment='ResNet34-UNet Architecture')
    dot.attr(rankdir='LR')
    
    # Input
    dot.node('input', 'Input\n(3 channels)\n256x256', shape='box')
    
    # Initial convolution
    dot.node('init_conv', 'Conv2d(7x7)\nBatchNorm2d\nReLU\nMaxPool2d\n(64 channels)\n64x64', shape='box')
    dot.edge('input', 'init_conv')
    
    # Downward path with ResNet blocks
    down_channels = [64, 64, 128, 256, 512]
    blocks = [3, 4, 6, 3]
    sizes = [64, 32, 16, 8]  # Sizes after each downsampling
    prev_node = 'init_conv'
    
    for i, (channels, num_blocks, size) in enumerate(zip(down_channels[1:], blocks, sizes)):
        node_name = f'down_{i}'
        dot.node(node_name, f'ResNet Block\n{num_blocks} blocks\n{channels} channels\n{size}x{size}', shape='box')
        dot.edge(prev_node, node_name)
        prev_node = node_name
    
    # Middle
    dot.node('middle', 'ResNet Block\n1 block\n1024 channels\n4x4', shape='box')
    dot.edge(prev_node, 'middle')
    
    # Upward path
    up_channels = [512, 256, 128, 64]
    sizes = [8, 16, 32, 64]  # Sizes after each upsampling
    prev_node = 'middle'
    
    for i, (channels, size) in enumerate(zip(up_channels, sizes)):
        node_name = f'up_{i}'
        dot.node(node_name, f'UpConv\n{channels} channels\n{size}x{size}', shape='box')
        dot.edge(prev_node, node_name)
        if i < len(up_channels):  # Skip last upconv for skip connections
            dot.edge(f'down_{len(down_channels)-2-i}', node_name, style='dashed')
        prev_node = node_name
    dot.node('last_conv', 'ConvTranspose2d\n(2x2)\nBatchNorm2d\nReLU\nConvTranspose2d\n(2x2)\nBatchNorm2d\nReLU\n64 channels\n256x256', shape='box')
    dot.edge(prev_node, 'last_conv')

    dot.node('output_conv', 'Conv2d(1x1)\nSigmoid\n1 channel\n256x256', shape='box')
    dot.edge('last_conv', 'output_conv')
    # Output
    dot.node('output', 'Output\n(1 channel)\n256x256', shape='box')
    dot.edge('output_conv', 'output')
    
    # Save the diagram
    dot.render('images/resnet34_unet', format='png', cleanup=True)

if __name__ == '__main__':
    create_resnet34_unet_diagram() 