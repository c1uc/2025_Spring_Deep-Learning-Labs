from graphviz import Digraph

def create_unet_diagram():
    dot = Digraph(comment='UNet Architecture')
    dot.attr(rankdir='LR')
    
    # Input
    dot.node('input', 'Input\n(3 channels)\n256x256', shape='box')
    
    # Downward path
    down_channels = [64, 128, 256, 512]
    sizes = [256, 128, 64, 32]  # Sizes after each downsampling
    prev_node = 'input'
    
    for i, (channels, size) in enumerate(zip(down_channels, sizes)):
        node_name = f'down_{i}'
        dot.node(node_name, f'DownConv\n{channels} channels\nMaxPool2d\n{size}x{size}', shape='box')
        dot.edge(prev_node, node_name)
        prev_node = node_name
    
    # Middle
    dot.node('middle', 'DoubleConv\n1024 channels\nMaxPool2d\n16x16', shape='box')
    dot.edge(prev_node, 'middle')
    
    # Upward path
    up_channels = [512, 256, 128, 64]
    sizes = [32, 64, 128, 256]  # Sizes after each upsampling
    prev_node = 'middle'
    
    for i, (channels, size) in enumerate(zip(up_channels, sizes)):
        node_name = f'up_{i}'
        dot.node(node_name, f'UpConv\n{channels} channels\n{size}x{size}', shape='box')
        dot.edge(prev_node, node_name)
        if i < len(up_channels):  # Skip last upconv for skip connections
            dot.edge(f'down_{len(down_channels)-1-i}', node_name, style='dashed')
        prev_node = node_name
    
    dot.node('output_conv', 'Conv2d(1x1)\nSigmoid\n1 channel\n256x256', shape='box')
    dot.edge(prev_node, 'output_conv')
    # Output
    dot.node('output', 'Output\n(1 channel)\n256x256', shape='box')
    dot.edge('output_conv', 'output')
    
    # Save the diagram
    dot.render('images/unet', format='png', cleanup=True)

if __name__ == '__main__':
    create_unet_diagram() 