import java.io.IOException;

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

public class TermFrequency extends Configured implements Tool {

	private static final Logger LOG = Logger.getLogger(TermFrequency.class);

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new TermFrequency(), args);
		System.exit(res);
	}

	public int run(String[] args) throws Exception {

		FileSystem fs = FileSystem.get(getConf());

		Path InputFilepath = new Path(args[0]);

		Path OutputPath = new Path(args[1]);
		if (fs.exists(OutputPath)) {
			fs.delete(OutputPath, true);
		}

		Job job = Job.getInstance(getConf(), "TermFreq");
		job.setJarByClass(TermFrequency.class);

		FileInputFormat.setInputDirRecursive(job, true);
		FileInputFormat.addInputPath(job, InputFilepath);
		FileOutputFormat.setOutputPath(job, OutputPath);
		job.setMapperClass(Map.class);
		job.setReducerClass(Reduce.class);

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);


		return job.waitForCompletion(true) ? 0 : 1;
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

			Text currentWord = new Text();
			for (String word : line.split("\\s+")) {
				if (word.isEmpty()) {
					continue;
				}

				currentWord = new Text(word + " " + folderName + "." + fileName.replace(".txt", ""));
				context.write(currentWord, one);
			}
		}
	}

	public static class Reduce extends
			Reducer<Text, IntWritable, Text, DoubleWritable> {
		public double totalWordCount = 0;
		public void reduce(Text word, Iterable<IntWritable> counts,
						   Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable count : counts) {
				sum += count.get();
			}
			context.write(word, new DoubleWritable(sum));
			}
		}
	}

