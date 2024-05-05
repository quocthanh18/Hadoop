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

public class DocClus {

    static double currentObjective = 0;
    public static String getTerms() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader("E:\\KHMT2-07\\input\\task_2.2\\terms.txt"));
        String line = br.readLine();
        br.close();
        return line;
    }

    public static class KMeansMapper extends Mapper<LongWritable, Text, Text, Text> {
        //        List<String> centroids = new ArrayList<String>();
        Map<String, OpenMapRealVector> centroids = new HashMap<String, OpenMapRealVector>();
        public void setup(Context context) throws IOException {
            Path init_pt = new Path("E:/doc_centroids.txt");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader br = null;
            br = new BufferedReader(new InputStreamReader(fs.open(init_pt)));
            String line;
            while ((line = br.readLine()) != null) {
                String cluster_ID = line.split("\\s+")[0];
                RealVector vector = new OpenMapRealVector(Arrays.stream(line.split("\\s+"), 1, line.split("\\s+").length)
                        .mapToDouble(Double::parseDouble).parallel().toArray());
                centroids.put(cluster_ID, (OpenMapRealVector) vector);
            }
            br.close();
            fs.close();
        }

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] lineParts = line.split("\\s+");
            String docID = lineParts[0];
            RealVector vector = new OpenMapRealVector(Arrays.stream(lineParts, 1, lineParts.length)
                    .mapToDouble(Double::parseDouble).parallel().toArray());
//            for (int i = 0; i < lineParts.length - 1; i++)
//                vector.setEntry(i, Double.parseDouble(lineParts[i + 1]));

            String centroid = "";
            double objective = 0;
            double max_similarity = Double.NEGATIVE_INFINITY;
            for (Map.Entry<String, OpenMapRealVector> entry : centroids.entrySet()) {
                String cluster_ID = entry.getKey();
                double similarity = vector.cosine(entry.getValue());
                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    centroid = cluster_ID;
                    objective = entry.getValue().getDistance(vector);
                }
            }
            vector = vector.append(1);
            vector = vector.append(objective);
            context.write(new Text(centroid), new Text(array2string(vector.toArray())));
//            context.write(new Text(centroid), new Text(docID + " " + array2string(vector.toArray()) + " " + 1 + " " + objective));
            int iteration = context.getConfiguration().getInt("N", 0);
            if ( iteration == 0 ) {
                BufferedWriter fw = new BufferedWriter(new FileWriter("E:/task_2_2.classes", true));
                fw.write("Cluster:  " + centroid + " | docID: " + docID + "\n");
                fw.close();
            }
        }

    } //centroid [vector] 1 objective

    public static class KMeansCombiner extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            RealVector sumVector = new OpenMapRealVector();
            int count = 0;
            double objective = 0;
            for (Text val : values) {
                String[] parts = val.toString().split("\\s+");
                int lengths = parts.length;
                count += Double.parseDouble(parts[lengths - 2]);
                objective += Double.parseDouble(parts[lengths - 1]);
                if (sumVector.getDimension() == 0)
                    sumVector = new OpenMapRealVector(parts.length - 2);
                sumVector = sumVector.add(new OpenMapRealVector(Arrays.stream(parts, 0, lengths - 2)
                        .mapToDouble(Double::parseDouble).parallel().toArray()));
//                for (int i = 0; i < parts.length - 3; i++)
//                    sumVector.setEntry(i, sumVector.getEntry(i) + Double.parseDouble(parts[i + 1]));
            }
//            context.write(key, new Text(array2string(sumVector.toArray()) + " " + count + " " + objective));
//            sumVector.setEntry(sumVector.getDimension() - 1, objective);
//            sumVector.setEntry(sumVector.getDimension() - 2, count);
            sumVector = sumVector.append(count);
            sumVector = sumVector.append(objective);
            context.write(key, new Text(array2string(sumVector.toArray())));
        }
    }


    public static class KMeansReducer extends Reducer<Text, Text, Text, Text> {

        BufferedWriter fw;
        public static enum Objective {
            Objective
        }
        double objective = 0;
        String terms;
        public void setup(Context context) throws IOException {
            int iteration = context.getConfiguration().getInt("N", 0);
            fw = new BufferedWriter(new FileWriter("E:/task_2_2.txt", true));
            fw.write("Iteration " + iteration + "\n");
            terms = context.getConfiguration().get("terms");
        }

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            RealVector sumVector = new OpenMapRealVector();
            int count = 0;
            for (Text val : values) { //[vector] count objective
                String [] parts = val.toString().split("\\s+");
                int lengths = parts.length;
                count += Double.parseDouble(parts[lengths - 2]);
                objective += Double.parseDouble(parts[lengths - 1]);
                if ( sumVector.getDimension() == 0 )
                    sumVector = new OpenMapRealVector(lengths - 2);
//                for (int i = 0; i < lengths - 2; i++)
//                    sumVector.setEntry(i, sumVector.getEntry(i) + Double.parseDouble(parts[i]));
                sumVector = sumVector.add(new OpenMapRealVector(Arrays.stream(parts, 0, parts.length - 2)
                        .mapToDouble(Double::parseDouble).parallel().toArray()));
            }
            sumVector = sumVector.mapDivide(count);
            context.write(key, new Text(array2string(sumVector.toArray())));
            int[] top10 = getTopKIndices(sumVector, 10);
            fw.write("Cluster " + key + " top 10 terms: ");
            for (int j : top10)
                fw.write(terms.split("\\s+")[j] + " ");
            fw.write("\n");
            int iteration = context.getConfiguration().getInt("N", 0);
            if (iteration == 0) {
                BufferedWriter cluster = new BufferedWriter(new FileWriter("E:/task_2_2.clusters", true));
                cluster.write(key + " " + array2string(sumVector.toArray()) + "\n");
                cluster.close();
            }
        }

        public void cleanup(Context context) throws IOException {
            int iteration = context.getConfiguration().getInt("N", 0);
            fw.write("End of Iteration " + iteration + "\n");
            fw.close();
            BufferedWriter loss = new BufferedWriter(new FileWriter("E:/ task_2_2.loss", true));
            loss.write("Iteration " + iteration + ": " + objective + "\n");
            loss.close();
            context.getCounter(Objective.Objective).setValue((long) objective);
        }
    }
    public static void main(String[] args) throws Exception {
        centroid_init("E:/doc_centroids.txt", new Integer(args[3]));
        int iterate = new Integer(args[2]);
        Configuration con = new Configuration();
        con.set("terms", getTerms());
        double prevObjective = Double.MAX_VALUE;
        while (true) {
            if (iterate == 0)
                break;
            iterate--;
            Job job = Job.getInstance(con, "KMeans");
            job.getConfiguration().setInt("N", iterate);
            job.setJarByClass(DocClus.class);
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
                currentObjective = job.getCounters().findCounter(KMeansReducer.Objective.Objective).getValue();
            } else
                System.exit(1);
            fs.delete(new Path(args[1]), true);
            fs.close();
//            if (Math.abs(prevObjective - currentObjective) < 0.5)
//                break;
//            prevObjective = currentObjective;
        }
    }


    public static void centroid_init(String path, int k) throws IOException {
        BufferedReader fr = new BufferedReader(new FileReader("E:\\KHMT2-07\\input\\task_2.2\\data.csv"));
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

        BufferedWriter fw = new BufferedWriter(new FileWriter(path, false));
        for (int i = 0; i < k; i++)
            fw.write(centroids[i] + "\n");
        fw.close();
    }


    public static String array2string(double[] array) {
        StringBuilder sb = new StringBuilder();
        for ( int i = 0; i < array.length - 1; i++)
            sb.append(array[i]).append(" ");
        sb.append(array[array.length - 1]);
        return sb.toString();
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

