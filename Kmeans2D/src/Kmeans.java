import java.util.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Kmeans {
    public static class KMeansMapper extends Mapper<LongWritable,
            Text,
            Text,
            Text>  {
        List <Double[]> centroids = new ArrayList<Double[]>();
        //Setup Function
        @Override
        protected void setup(Context context) throws IOException {
            Path init_pt = new Path("E:/centroids.txt");
            Configuration config = new Configuration();
            FileSystem fs = FileSystem.get(new Configuration());
            BufferedReader br = null;
            br = new BufferedReader(new InputStreamReader(fs.open(init_pt)));
            String line;
            while ((line = br.readLine()) != null) {
                String [] parts = line.split("\\s+");
                double x = Double.parseDouble(parts[1]);
                double y = Double.parseDouble(parts[2]);
                centroids.add(new Double[]{x,y});
            }
            br.close();
            fs.close();
        }
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String [] parts = line.split(",");
            double x = Double.parseDouble(parts[1]);
            double y = Double.parseDouble(parts[2]);
            int centroid = 0;
            double min_dist = Double.MAX_VALUE;
            for(int i = 0; i < centroids.size(); i++) {
                double distance = Math.sqrt(Math.pow((x - centroids.get(i)[0]), 2) + Math.pow((y - centroids.get(i)[1]), 2));
                if(distance < min_dist) {
                    min_dist = distance;
                    centroid = i;
                }
            }
        context.write(new Text(String.valueOf(centroid)), new Text(String.valueOf(x) + " " + String.valueOf(y)));
        }
    }
    public static class KMeansReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double sum_x = 0;
            double sum_y = 0;
            int count = 0;
            for (Text val : values) {
                String [] parts = val.toString().split(" ");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                sum_x += x;
                sum_y += y;
                count ++;
            }
            double new_x = sum_x / count;
            double new_y = sum_y / count;
            context.write(key, new Text(String.valueOf(new_x) + " " + String.valueOf(new_y)));
        }
    }
    public static void main(String[] args) throws Exception {
        List <Double[]> prevCentorids = new ArrayList<Double[]>();
        List <Double[]> currCentorids = new ArrayList<Double[]>();
        // Mapper class
        int iterations = 20;
        int iterate = iterations;
        while (true) {
            if (iterate == 0)
                break;
            --iterate;
            Configuration con = new Configuration();
            Job job = Job.getInstance(con, "KMeans");
            job.setJarByClass(Kmeans.class);
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);
            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(Text.class);
            FileSystem fs = FileSystem.get(con);
            FileInputFormat.addInputPath(job, new Path(args[0]));
            FileOutputFormat.setOutputPath(job, new Path(args[1]));
            if (job.waitForCompletion(true)) {
                FileUtil.copy(fs, new Path(args[1] + "/part-r-00000"), fs, new Path("E:/centroids.txt"),
                        true, true, con);
            } else
                System.exit(1);
            prevCentorids = currCentorids;
            currCentorids = new ArrayList<Double[]>();
            BufferedReader br = null;
            br = new BufferedReader(new InputStreamReader(fs.open(new Path( "E:/centroids.txt"))));
            String line;
            while ((line = br.readLine()) != null) {
                String [] parts = line.split("\\s+");
                double x = Double.parseDouble(parts[1]);
                double y = Double.parseDouble(parts[2]);
                currCentorids.add(new Double[]{x,y});
            }
            br.close();
            if (checkSimilar(prevCentorids, currCentorids)){
                fs.delete(new Path(args[1]), true);
                fs.close();
                break;
            }
            fs.delete(new Path(args[1]), true);
            fs.close();
        }
    }
    public static boolean checkSimilar(List<Double[]> prevCentorids, List<Double[]> currCentorids) {
                double changes = 0;
                for(int i = 0; i < prevCentorids.size(); i++)
                    changes += Math.sqrt(Math.pow((prevCentorids.get(i)[0] - currCentorids.get(i)[0]), 2) + Math.pow((prevCentorids.get(i)[1] - currCentorids.get(i)[1]), 2));
        return changes < 0.8;
    }
}
