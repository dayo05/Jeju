using System.ComponentModel;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using static TorchSharp.torch;
using static TorchSharp.torchvision.models;
using static TorchSharp.torch.utils.data;
using static System.Linq.Enumerable;

var trainDataset = new JejuDataset(true);
var testDataset = new JejuDataset(false);
var train = new DataLoader(trainDataset, device: CUDA, num_worker: 5, batchSize: 32);
var validation = new DataLoader(testDataset, device: CUDA, num_worker: 10, batchSize: 64);

var model = resnet34(6, device: CUDA);
var criterion = nn.functional.cross_entropy_loss();
var optimizer = optim.Adam(model.parameters(), lr: 0.001);

foreach(var x in Range(0, 1000)) {
    var avg_cost = 0.0;
    foreach (var t in train)
    {
        optimizer.zero_grad();
        var hypothesis = model.forward(t["image"]);
        var cost = criterion(hypothesis, t["label"]);
        cost.backward();
        optimizer.step();

        avg_cost += cost.cpu().item<float>() / train.Count;
    }
    Console.Write($"{avg_cost}, ");

    avg_cost = 0;
    using (no_grad())
    {
        foreach (var t in validation)
        {
            var hypothesis = model.forward(t["image"]);
            avg_cost += criterion(hypothesis, t["label"]).cpu().item<float>() / validation.Count;
        }
    }
    Console.WriteLine(avg_cost);
}



class JejuDataset : Dataset
{
    private List<string> files;
    public JejuDataset(bool isTrain)
    {
        files = Directory.GetDirectories("/home/dayo/datasets/jeju" + (isTrain ? "/train/jeju" : "/test/jeju") ).SelectMany(Directory.GetFiles).Where(x => x.EndsWith(".png")).ToList();
    }

    public override Dictionary<string, Tensor> GetTensor(long index) => new Dictionary<string, Tensor>
    {
        { "label", tensor(long.Parse(files[(int)index].Split(Path.DirectorySeparatorChar)[^2]) - 1) },
        { "image", ReadImage(files[(int)index]) }
    };
    
    public override long Count => files.Count;
    
    Tensor ReadImage(string image)
    {
        var img = Image.Load<Rgb24>(image, new PngDecoder());
        img.Mutate(x => x.Resize(224, 224));
        return cat(new[] {ReadImageChannel(img, "R"), ReadImageChannel(img, "G"), ReadImageChannel(img, "B")}, 0);
    }

    private Tensor ReadImageChannel(Image<Rgb24> image, string channel)
        => tensor(image.GetPixelMemoryGroup()[0].Span.ToArray().Select(x => channel switch
            {
                "R" => x.R,
                "G" => x.G,
                "B" => x.B,
                _ => throw new InvalidEnumArgumentException("not defined channel")} / 255.0f).ToList(),
            new long[]{1, image.Width, image.Height});
}
