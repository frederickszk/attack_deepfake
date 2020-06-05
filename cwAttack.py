import sys
import tensorflow as tf
import numpy as np
import tensorflow.keras as K

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess


class CW:
    def __init__(self, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY,
                 initial_const = INITIAL_CONST,
                 boxmin = 0, boxmax = 1):

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.repeat = binary_search_steps >= 10
        self.batch_size = batch_size

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False


        self.model=model

        # Assert that model input shape is (None, width, width, channels)
        image_size, num_channels = model.input.shape[1], model.input.shape[3]
        self.shape = (batch_size, image_size, image_size, num_channels)
        

        # Optimize target
        self.modifier = tf.Variable(np.zeros(self.shape, dtype=np.float32))

        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.optimizer = K.optimizers.Adam(self.LEARNING_RATE)

    def initState(self):
        self.modifier.assign(np.zeros(self.shape, dtype=np.float32))
        self.optimizer = K.optimizers.Adam(self.LEARNING_RATE)


    @tf.function
    def train_step(self, timg, tlab, const):
        '''
        :param timg: Original images batch.
        :param tlab: Target/True labels batch.
        :param const: Binary searched const C.

        :return:loss, L2dis, model_output(scores), new_image
        '''
        with tf.GradientTape() as tape:
            tape.watch(self.modifier)
            # print(type(self.modifier))
            newimg = tf.tanh(self.modifier + timg) * self.boxmul + self.boxplus
            output = self.model(newimg)
            real = tf.reduce_sum(tlab * output, 1)
            other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)
            if self.TARGETED:
                # if targetted, optimize for making the other class most likely
                loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
            else:
                # if untargeted, optimize for making this class least likely.
                loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)
            l2dist = tf.reduce_sum(tf.square(newimg-(tf.tanh(timg) * self.boxmul + self.boxplus)), [1, 2, 3])
            loss1 = tf.reduce_sum(const * loss1)
            loss2 = tf.reduce_sum(l2dist)
            loss = loss1 + loss2
        gradients = tape.gradient(loss, self.modifier)
        self.optimizer.apply_gradients([(gradients, self.modifier)])
        return loss, l2dist, output, newimg

    def attack(self, imgs, targets):
        l2s = []
        scs = []
        ats = []
        for i in range(0, len(imgs), self.batch_size):
            l2, sc, at = self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size])
            l2s.extend(l2)
            scs.extend(sc)
            ats.extend(at)
        return np.array(l2s), np.array(scs), np.array(ats)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = tf.cast(np.ones(batch_size) * self.initial_const, tf.float32)
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.initState()

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                # CONST = upper_bound
                CONST = tf.cast(upper_bound, tf.float32)

            prev = np.inf
            for iteration in range(self.MAX_ITERATIONS):

                loss, l2s, scores, nimg = self.train_step(batch, batchlab, CONST)

                if np.all(scores >= -.0001) and np.all(scores <= 1.0001):
                    if np.allclose(np.sum(scores, axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception(
                                "The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration != 0 and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if loss > prev * .9999:
                        break
                    prev = loss


                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2.numpy()
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2.numpy()
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii.numpy()

            # adjust the constant as needed
            const_tmp = []
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
#                     print(CONST[e])
                    if upper_bound[e] < 1e9:
                        const_tmp.append((lower_bound[e] + upper_bound[e]) / 2)
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        const_tmp.append((lower_bound[e] + upper_bound[e]) / 2)
                    else:
                        const_tmp.append(CONST[e] * 10)
            CONST = tf.cast(const_tmp, tf.float32)

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestl2, o_bestscore, o_bestattack