import java.io.IOException;

import java.util.HashMap;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.log4j.Logger;

public class HighestAverage extends Configured implements Tool {

    private static final Logger LOG = Logger.getLogger(HighestAverage.class);

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new HighestAverage(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {

        FileSystem fs = FileSystem.get(getConf());

        Path InputFilepath = new Path(args[0]);
        Path OutputPath = new Path(args[1]);
        if (fs.exists(OutputPath)) {
            fs.delete(OutputPath, true);
        }
        Job job = Job.getInstance(getConf(), "HighAvg");
        job.setJarByClass(HighestAverage.class);

        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, InputFilepath);
        FileOutputFormat.setOutputPath(job, OutputPath);
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        if (!job.waitForCompletion(true)) {
            return 1;
        }
        Job job2 = Job.getInstance(getConf(), "Top5");
        job2.setJarByClass(HighestAverage.class);
        FileInputFormat.addInputPath(job2, OutputPath);
        FileOutputFormat.setOutputPath(job2, new Path(args[2]));
        job2.setMapperClass(Map2.class);
        job2.setReducerClass(Reduce2.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        return job2.waitForCompletion(true) ? 0 : 1;

    }

    public static class Map extends
            Mapper<LongWritable, Text, Text, DoubleWritable> {
        private final static IntWritable one = new IntWritable(1);

        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            String line = lineText.toString(); // term doc freq
            String[] fields = line.split(" ");
            Text currentWord = new Text(fields[0]);
            Text class_ = new Text(fields[1].split("\\.")[0]);
            double freq = Double.parseDouble(fields[2]);
            context.write(new Text(currentWord + " " + class_), new DoubleWritable(freq));
        }
    }

    public static class Reduce extends
            Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        public void reduce(Text word, Iterable<DoubleWritable> counts,
                           Context context) throws IOException, InterruptedException {
            double sum = 0;
            int counter = 0;
            for (DoubleWritable count : counts) {
                sum += count.get();
                counter++;
            }
            double avg = sum / counter;
            context.write(word, new DoubleWritable(avg));
        }
    }

    public static class Map2 extends
            Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            String line = lineText.toString(); // term doc freq
            String[] fields = line.split("\\s");
            String class_ = fields[1];
            String word = fields[0];
            String freq = fields[2];
            context.write(new Text(class_), new Text(word + " " + freq));
        }
    }

    public static class Reduce2 extends
            Reducer<Text, Text, Text, Text> {
        public void reduce(Text word, Iterable<Text> counts,
                           Context context) throws IOException, InterruptedException {
            TreeMap<Double, String> map = new TreeMap<Double, String>();
            for (Text count : counts) {
                String[] fields = count.toString().split(" ");
                String word_ = fields[0];
                double freq = Double.parseDouble(fields[1]);
                map.put(freq, word_);
                if (map.size() > 5) {
                    map.remove(map.firstKey());
                }
            }
            for (java.util.Map.Entry<Double, String> entry : map.entrySet()) {
                context.write(word, new Text(entry.getValue() + " " + entry.getKey()));
            }
        }
    }
}
