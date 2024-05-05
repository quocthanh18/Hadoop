import java.io.*;
import java.util.*;
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
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.OpenMapRealVector;
public class test {
    public static class KMeansMapper extends Mapper<LongWritable, Text, Text, Text> {
        //        List<String> centroids = new ArrayList<String>();
        Map<String, OpenMapRealVector> centroids = new HashMap<String, OpenMapRealVector>();

        public void setup(Context context) throws IOException {
            Path init_pt = new Path("E:/doc_centroids.txt");
            FileSystem fs = FileSystem.get(new Configuration());
            BufferedReader br = null;
            br = new BufferedReader(new InputStreamReader(fs.open(init_pt)));
            String line;
            while ((line = br.readLine()) != null) {
                String cluster_ID = line.split("\\s+")[0];
                RealVector vector = new OpenMapRealVector(line.split("\\s+").length - 1);
                for (int i = 0; i < line.split("\\s+").length - 1; i++)
                    vector.setEntry(i, Double.parseDouble(line.split("\\s+")[i + 1].split(":")[1]));
                centroids.put(cluster_ID, (OpenMapRealVector) vector);
            }
            br.close();
            fs.close();
        }

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] lineParts = line.split("\\s+");
            String docID = lineParts[0];
            RealVector vector = new OpenMapRealVector(lineParts.length - 1);
            String[] terms_vector = new String[lineParts.length - 1];
            for (int i = 0; i < lineParts.length - 1; i++) {
                vector.setEntry(i, Double.parseDouble(lineParts[i + 1].split(":")[1]));
                terms_vector[i] = lineParts[i + 1];
            }
            String centroid = "";
            double objective = 0;
            double min_dist = Double.MAX_VALUE;
            for (Map.Entry<String, OpenMapRealVector> entry : centroids.entrySet()) {
                String cluster_ID = entry.getKey();
                double distance = vector.cosine(entry.getValue());
                if (distance < min_dist) {
                    min_dist = distance;
                    centroid = cluster_ID;
                    objective = entry.getValue().getDistance(vector);
                }
            }
            context.write(new Text(centroid), new Text(vector2string(terms_vector) + " " + 1 + " " + objective));
            int iteration = context.getConfiguration().getInt("N", 0);
            if (iteration == 0){
                FileWriter cluster = new FileWriter("E:/task_2_2_classes.txt", true);
                cluster.write(centroid + " " + docID + "\n");
                cluster.close();
            }
        }

    }

    public static class KMeansCombiner extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            RealVector sumVector = new OpenMapRealVector();
            int count = 0;
            double objective = 0;
            for (Text val : values) {   //centroid [term:vector] 1 objective
                String [] parts = val.toString().split("\\s+");
                count += Integer.parseInt(parts[parts.length - 2]);
                objective += Double.parseDouble(parts[parts.length - 1]);
                if ( sumVector.getDimension() == 0 ) {
                    sumVector = new OpenMapRealVector(parts.length - 2);
                }
                for (int i = 0; i < parts.length - 2; i++) {
                    sumVector.setEntry(i, sumVector.getEntry(i) + Double.parseDouble(parts[i].split(":")[1]));
                }
            }
            String [] value = new String[sumVector.getDimension()];
            for (int i = 0; i < sumVector.getDimension(); i++)
                value[i] = key.toString() + ":" + sumVector.getEntry(i);
            context.write(key, new Text(vector2string(value) + " " + count + " " + objective));
        }
    }
    public static class KMeansReducer extends Reducer<Text, Text, Text, Text> {

        FileWriter fw;
        double objective = 0;
        public void setup(Context context) throws IOException {
            int iteration = context.getConfiguration().getInt("N", 0);
            fw = new FileWriter("E:/task_2_2.txt", true);
            fw.write("Iteration " + iteration + "\n");
        }

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            RealVector sumVector = new OpenMapRealVector();
            String[] termVector = null;  //centroid [term:vector] count objective
            int count = 0;
            for (Text val : values) {
                String [] parts = val.toString().split("\\s+");
                int lengths = parts.length;
                count += Integer.parseInt(parts[lengths - 2]);
                objective += Double.parseDouble(parts[lengths - 1]);
                if ( sumVector.getDimension() == 0 ) {
                    sumVector = new OpenMapRealVector(lengths - 2);
                    termVector = new String[lengths - 2];
                }
                for (int i = 0; i < lengths - 2; i++) {
                    sumVector.setEntry(i, sumVector.getEntry(i) + Double.parseDouble(parts[i].split(":")[1]));
                    termVector[i] = parts[i].split(":")[0];
                }
            }
            sumVector = sumVector.mapDivide(count);
            String [] value = new String[sumVector.getDimension()];
            for (int i = 0; i < sumVector.getDimension(); i++)
                value[i] = termVector[i] + ":" + sumVector.getEntry(i);
            context.write(key, new Text(vector2string(value)));
            int[] top10 = getTopKIndices(sumVector, 10);
            fw.write("Cluster " + key + " top 10 terms: ");
            for (int i = 0; i < top10.length; i++)
                fw.write(termVector[top10[i]] + " ");
            fw.write("\n");
            int iteration = context.getConfiguration().getInt("N", 0);
            if (iteration == 0){
                FileWriter cluster = new FileWriter("E:/task_2_2_cluster.txt", true);
                cluster.write(key + " " + vector2string(value) + "\n");
                cluster.close();
            }
        }

        public void cleanup(Context context) throws IOException {
            int iteration = context.getConfiguration().getInt("N", 0);
            fw.write("End of Iteration " + iteration + "\n");
            fw.close();
            FileWriter loss = new FileWriter("E:/task_2_2_loss.txt", true);
            loss.write("Iteration " + iteration + ": " + objective + "\n");
            loss.close();
        }
    }
    public static void main(String[] args) throws Exception {
        centroid_init("E:/doc_centroids.txt", 3);
        int iterate = 1;
        while (true) {
            if (iterate == 0)
                break;
            iterate--;
            Configuration con = new Configuration();
            con.set("mapreduce.input.fileinputformat.split.minsize", "134217728");
            Job job = Job.getInstance(con, "KMeans");
            job.getConfiguration().setInt("N", iterate);
            job.setJarByClass(test.class);
            job.setMapperClass(KMeansMapper.class);
            job.setCombinerClass(KMeansCombiner.class);
            job.setReducerClass(KMeansReducer.class);
            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(Text.class);
            FileSystem fs = FileSystem.get(con);
            FileInputFormat.addInputPath(job, new Path(args[0]));
            FileOutputFormat.setOutputPath(job, new Path(args[1]));
            if (job.waitForCompletion(true)) {
                FileUtil.copy(fs, new Path(args[1] + "/part-r-00000"), fs, new Path("E:/doc_centroids.txt"),
                        true, true, con);
            } else
                System.exit(1);
            fs.delete(new Path(args[1]), true);
            fs.close();
        }
    }


    public static void centroid_init(String path, int k) throws IOException {
        BufferedReader fr = new BufferedReader(new FileReader("E:/doc_term_matrix.csv"));
        String[] centroids = new String[k];
        List<String> temp = new ArrayList<String>();
        String line = fr.readLine();
        while (line != null) {
            temp.add(line);
            line = fr.readLine();
        }

        Collections.shuffle(temp);
        for (int i = 0; i < k; i++)
            centroids[i] = temp.get(i);

        fr.close();

        FileWriter fw = new FileWriter(path, false);
        for (int i = 0; i < k; i++)
            fw.write(centroids[i] + "\n");
        fw.close();
    }


    public static <T> String vector2string(T[] vector) {
        String vector_str = "";
        for (int i = 0; i < vector.length - 1; i++)
            vector_str += vector[i] + " ";
        vector_str += vector[vector.length - 1];
        return vector_str;
    }

    public static class IndexValuePair {
        int index;
        double value;

        public IndexValuePair(int index, double value) {
            this.index = index;
            this.value = value;
        }
    }

    public static int[] getTopKIndices(RealVector array, int num) {
        PriorityQueue<IndexValuePair> queue = new PriorityQueue<>(Comparator.comparingDouble((IndexValuePair value) -> value.value));

        for (int i = 0; i < array.getDimension(); i++) {
            queue.offer(new IndexValuePair(i, array.getEntry(i)));
            if (queue.size() > num) {
                queue.poll();
            }
        }

        int[] result = new int[num];
        for (int i = 0; i < num; i++) {
            result[num - 1 - i] = queue.poll().index;
        }
        return result;
    }
}

