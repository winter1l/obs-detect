Texture2D<float4> InputTexture : register(t0);
SamplerState BilinearSampler : register(s0);
RWStructuredBuffer<float> OutputBuffer : register(u0);

cbuffer PreprocessParams : register(b0)
{
    uint SrcWidth;
    uint SrcHeight;
    uint DstWidth;
    uint DstHeight;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x;
    uint y = dtid.y;

    if (x >= DstWidth || y >= DstHeight)
        return;

    // Calculate aspect-ratio preserving dimensions
    float r = min((float)DstWidth / (float)SrcWidth, (float)DstHeight / (float)SrcHeight);
    uint unpadWidth = (uint)round(r * (float)SrcWidth);
    uint unpadHeight = (uint)round(r * (float)SrcHeight);

    float3 pixel;

    if (x < unpadWidth && y < unpadHeight)
    {
        // OpenCV bilinear interpolation mapping: ((dst_coord + 0.5) * scale - 0.5)
        float scale_x = (float)SrcWidth / (float)unpadWidth;
        float scale_y = (float)SrcHeight / (float)unpadHeight;

        float src_x = ((float)x + 0.5f) * scale_x - 0.5f;
        float src_y = ((float)y + 0.5f) * scale_y - 0.5f;

        // Manual Bilinear Interpolation to match OpenCV perfectly and bypass driver sampler issues
        int x1 = (int)floor(src_x);
        int y1 = (int)floor(src_y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        float dx = src_x - (float)x1;
        float dy = src_y - (float)y1;

        // Boundary clamp (equivalent to BORDER_REPLICATE in OpenCV)
        x1 = clamp(x1, 0, (int)SrcWidth - 1);
        x2 = clamp(x2, 0, (int)SrcWidth - 1);
        y1 = clamp(y1, 0, (int)SrcHeight - 1);
        y2 = clamp(y2, 0, (int)SrcHeight - 1);

        float4 p11 = InputTexture[int2(x1, y1)];
        float4 p21 = InputTexture[int2(x2, y1)];
        float4 p12 = InputTexture[int2(x1, y2)];
        float4 p22 = InputTexture[int2(x2, y2)];

        float4 color = lerp(lerp(p11, p21, dx), lerp(p12, p22, dx), dy);
        pixel = color.rgb;
    }
    else
    {
        pixel = float3(114.0f / 255.0f, 114.0f / 255.0f, 114.0f / 255.0f);
    }

    uint channelSize = DstWidth * DstHeight;
    uint baseIndex = y * DstWidth + x;

    // Output is expected to be RGB order (NCHW), scaled to 0-255
    OutputBuffer[0 * channelSize + baseIndex] = pixel.r * 255.0f; // Red
    OutputBuffer[1 * channelSize + baseIndex] = pixel.g * 255.0f; // Green
    OutputBuffer[2 * channelSize + baseIndex] = pixel.b * 255.0f; // Blue
}
