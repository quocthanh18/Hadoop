import java.io.IOException;

import java.util.HashMap;
import java.util.Map;
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

public class TFIDF1 extends Configured implements Tool {

    private static final Logger LOG = Logger.getLogger(TFIDF1.class);

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new TFIDF1(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {

        FileSystem fs = FileSystem.get(getConf());

        Path InputFilepath = new Path(args[0]);
        Path OutputPath = new Path(args[1]);
        Path IntermediateOutputPath = new Path(args[2]);
        if (fs.exists(OutputPath)) {
            fs.delete(OutputPath, true);
        }
        if (fs.exists(IntermediateOutputPath)) {
            fs.delete(IntermediateOutputPath, true);
        }
        Job job = Job.getInstance(getConf(), "TFIDF1");
        job.setJarByClass(TFIDF1.class);

        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, InputFilepath);
        FileOutputFormat.setOutputPath(job, OutputPath);
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        if (!job.waitForCompletion(true)) {
            return 1;
        }
        Job job2 = Job.getInstance(getConf(), "TFIDF2");
        job2.setJarByClass(TFIDF1.class);
        job2.setMapperClass(IDF_Map.class);
        job2.setReducerClass(IDF_Reduce.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job2, OutputPath);
        FileOutputFormat.setOutputPath(job2, IntermediateOutputPath);
        return job2.waitForCompletion(true) ? 0 : 1;
    }

    public static class Map extends
            Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);

        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            String line = lineText.toString();
            Path filePath = ((FileSplit) context.getInputSplit()).getPath();
            String fileName = filePath.getName();
            String folderName = filePath.getParent().getName().toLowerCase();

            String docID = folderName + "." + fileName.replace(".txt", "");
            for (String word : line.split("\\s+")) {
                if (word.isEmpty()) {
                    continue;
                }
                context.write(new Text(docID + " " + word), one);
                context.write(new Text(docID), one);
            }
        }
    }

    public static class Reduce extends
            Reducer<Text, IntWritable, Text, DoubleWritable> {
        public int wordCount = 0;
        public void reduce(Text word, Iterable<IntWritable> counts,
                           Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable count : counts) {
                sum += count.get();
            }
            if (word.toString().contains(" ")) {
                String word_ = word.toString().split(" ")[1];
                String docID = word.toString().split(" ")[0];
                context.write(new Text(word_ + " " + docID), new DoubleWritable(sum / (double) wordCount));
            } else {
                wordCount = sum;
            }

        }
    }

    public static class IDF_Map extends
            Mapper<LongWritable, Text, Text, Text> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private Text docID = new Text();
        public void map(LongWritable offset, Text lineText, Context context) //107bn business.001 0.3838
                throws IOException, InterruptedException {
                String line = lineText.toString();
                String[] parts = line.split("\\s+");
                this.word.set(parts[0]);
                this.docID.set(parts[1] + " " + parts[2]);
                context.write(word, docID);
        }
    }

    public static class IDF_Reduce extends
            Reducer<Text, Text, Text, DoubleWritable> {
        private Text word_file = new Text();
        private double tfidf;
        public void reduce(Text word, Iterable<Text> counts,
                           Context context) throws IOException, InterruptedException {
        double docswithword = 0;
        java.util.Map<String, Double> wordCount = new HashMap<String, Double>();
        for (Text docID : counts) {
            String [] filecounter = docID.toString().split(" ");
            docswithword++;
            wordCount.put(filecounter[0], Double.parseDouble(filecounter[1]));
            }
        double idf = Math.log10(2225 / docswithword);
        for ( String temp_tfidf_file : wordCount.keySet()) {
            this.word_file.set(word + " " + temp_tfidf_file);
            this.tfidf = wordCount.get(temp_tfidf_file) * idf;
            context.write(this.word_file, new DoubleWritable(this.tfidf));
            }
        }
    }
}


