"""3D Geometric Algebra.

Created by  : https://bivector.net/tools.html?p=0&q=2&r=0

Quaternion algebra based on Clifford Algebra (0,2,0)
"""

from math import sqrt, pi, sin, cos, asin, acos, atan2, exp, log
from copy import deepcopy
import numpy as np # Numpy is required for many vector operations

class Quaternion:

    def __init__(self, *args, **kwargs):
        """Initialise a new Quaternion object.

        See Object Initialisation docs for complete behaviour:

        https://kieranwynn.github.io/pyquaternion/#object-initialisation

        """
        self.q      = [0] * 4
        self._base  = ["1", "e1", "e2", "e12"]

        s = len(args)
        if s == 0:
            # No positional arguments supplied
            if kwargs:
                # Keyword arguments provided
                if ("scalar" in kwargs) or ("vector" in kwargs):
                    scalar = kwargs.get("scalar", 0.0)
                    if scalar is None:
                        scalar = 0.0
                    else:
                        scalar = float(scalar)

                    vector = kwargs.get("vector", [])
                    vector = self._validate_number_sequence(vector, 3)

                    self.q = np.hstack((scalar, vector))
                elif ("real" in kwargs) or ("imaginary" in kwargs):
                    real = kwargs.get("real", 0.0)
                    if real is None:
                        real = 0.0
                    else:
                        real = float(real)

                    imaginary = kwargs.get("imaginary", [])
                    imaginary = self._validate_number_sequence(imaginary, 3)

                    self.q = np.hstack((real, imaginary))
                elif ("axis" in kwargs) or ("radians" in kwargs) or ("degrees" in kwargs) or ("angle" in kwargs):
                    try:
                        axis = self._validate_number_sequence(kwargs["axis"], 3)
                    except KeyError:
                        raise ValueError(
                            "A valid rotation 'axis' parameter must be provided to describe a meaningful rotation."
                        )
                    angle = kwargs.get('radians') or self.to_radians(kwargs.get('degrees')) or kwargs.get('angle') or 0.0
                    self.q = Quaternion._from_axis_angle(axis, angle).q
                elif "array" in kwargs:
                    self.q = self._validate_number_sequence(kwargs["array"], 4)
                elif "matrix" in kwargs:
                    optional_args = {key: kwargs[key] for key in kwargs if key in ['rtol', 'atol']}
                    self.q = Quaternion._from_matrix(kwargs["matrix"], **optional_args).q
                else:
                    keys = sorted(kwargs.keys())
                    elements = [kwargs[kw] for kw in keys]
                    if len(elements) == 1:
                        r = float(elements[0])
                        self.q = np.array([r, 0.0, 0.0, 0.0])
                    else:
                        self.q = self._validate_number_sequence(elements, 4)

            else:
                # Default initialisation
                self.q = np.array([1.0, 0.0, 0.0, 0.0])
        elif s == 1:
            # Single positional argument supplied
            if isinstance(args[0], Quaternion):
                self.q = args[0].q
                return
            if args[0] is None:
                raise TypeError("Object cannot be initialised from {}".format(type(args[0])))
            try:
                r = float(args[0])
                self.q = np.array([r, 0.0, 0.0, 0.0])
                return
            except TypeError:
                pass  # If the single argument is not scalar, it should be a sequence

            self.q = self._validate_number_sequence(args[0], 4)
            return
        
        elif s == 2:  # scalar and vector
            scalar = float(args[0])
            vector = args[1]
            vector = self._validate_number_sequence(vector, 3)
            self.q = np.hstack((scalar, vector))

        else:
            # More than one positional argument supplied
            self.q = self._validate_number_sequence(args, 4)


    # def __init__(self, value=0, index=0):
    #     """Initiate a new Quaternion.
         
    #     Optional, the component index can be set with value.
    #     """
    #     self.q  = [0] * 4
    #     self._base = ["1", "e1", "e2", "e12"]
    #     if (value != 0):
    #         self.q[index] = value
            
    @classmethod
    def fromarray(cls, array):
        """Initiate a new Quaternion from an array-like object.

        The first axis of the array is assumed to correspond to the elements
        of the algebra, and needs to have the same length. Any other dimensions
        are left unchanged, and should have simple operations such as addition 
        and multiplication defined. NumPy arrays are therefore a perfect 
        candidate. 

        :param array: array-like object whose length is the dimension of the algebra.
        :return: new instance of Quaternion.
        """
        self = cls()
        if len(array) != len(self):
            raise TypeError('length of array must be identical to the dimension '
                            'of the algebra.')
        self.q = array
        return self

      # Initialise from axis-angle
    
    @classmethod
    def _from_axis_angle(cls, axis, angle):
        """Initialise from axis and angle representation

        Create a Quaternion by specifying the 3-vector rotation axis and rotation
        angle (in radians) from which the quaternion's rotation should be created.

        Params:
            axis: a valid numpy 3-vector
            angle: a real valued angle in radians
        """
        mag_sq = np.dot(axis, axis)
        if mag_sq == 0.0:
            raise ZeroDivisionError("Provided rotation axis has no length")
        # Ensure axis is in unit vector form
        if (abs(1.0 - mag_sq) > 1e-12):
            axis = axis / sqrt(mag_sq)
        theta = angle / 2.0
        r = cos(theta)
        i = axis * sin(theta)

        return cls.fromarray([r, i[0], i[1], i[2]])

    @classmethod
    def random(cls):
        """Generate a random unit quaternion.

        Uniformly distributed across the rotation space
        As per: http://planning.cs.uiuc.edu/node198.html
        """
        r1, r2, r3 = np.random.random(3)

        q1 = sqrt(1.0 - r1) * (sin(2 * pi * r2))
        q2 = sqrt(1.0 - r1) * (cos(2 * pi * r2))
        q3 = sqrt(r1)       * (sin(2 * pi * r3))
        q4 = sqrt(r1)       * (cos(2 * pi * r3))

        return cls.fromarray([q1, q2, q3, q4])   

    @classmethod
    def exp(cls, q):
        """Quaternion Exponential.

        Find the exponential of a quaternion amount.

        Params:
             q: the input quaternion/argument as a Quaternion object.

        Returns:
             A quaternion amount representing the exp(q). See [Source](https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation for more information and mathematical background).

        Note:
             The method can compute the exponential of any quaternion.
        """
        tolerance = 1e-17
        v_norm = np.linalg.norm(q.vector)
        vec = q.vector
        if v_norm > tolerance:
            vec = vec / v_norm
        magnitude = exp(q.scalar)
        return Quaternion(scalar = magnitude * cos(v_norm), vector = magnitude * sin(v_norm) * vec)

    @classmethod
    def log(cls, q):
        """Quaternion Logarithm.

        Find the logarithm of a quaternion amount.

        Params:
             q: the input quaternion/argument as a Quaternion object.

        Returns:
             A quaternion amount representing log(q) := (log(|q|), v/|v|acos(w/|q|)).

        Note:
            The method computes the logarithm of general quaternions. See [Source](https://math.stackexchange.com/questions/2552/the-logarithm-of-quaternion/2554#2554) for more details.
        """
        v_norm = np.linalg.norm(q.vector)
        q_norm = q.norm
        tolerance = 1e-17
        if q_norm < tolerance:
            # 0 quaternion - undefined
            return Quaternion(scalar=-float('inf'), vector=float('nan')*q.vector)
        if v_norm < tolerance:
            # real quaternions - no imaginary part
            return Quaternion(scalar=log(q_norm), vector=[0, 0, 0])
        vec = q.vector / v_norm
        return Quaternion(scalar=log(q_norm), vector=acos(q.scalar/q_norm)*vec)

    @classmethod
    def exp_map(cls, q, eta):
        """Quaternion exponential map.

        Find the exponential map on the Riemannian manifold described
        by the quaternion space.

        Params:
             q: the base point of the exponential map, i.e. a Quaternion object
           eta: the argument of the exponential map, a tangent vector, i.e. a Quaternion object

        Returns:
            A quaternion p such that p is the endpoint of the geodesic starting at q
            in the direction of eta, having the length equal to the magnitude of eta.

        Note:
            The exponential map plays an important role in integrating orientation
            variations (e.g. angular velocities). This is done by projecting
            quaternion tangent vectors onto the quaternion manifold.
        """
        return q * Quaternion.exp(eta)

    @classmethod
    def sym_exp_map(cls, q, eta):
        """Quaternion symmetrized exponential map.

        Find the symmetrized exponential map on the quaternion Riemannian
        manifold.

        Params:
             q: the base point as a Quaternion object
           eta: the tangent vector argument of the exponential map
                as a Quaternion object

        Returns:
            A quaternion p.

        Note:
            The symmetrized exponential formulation is akin to the exponential
            formulation for symmetric positive definite tensors [Source](http://www.academia.edu/7656761/On_the_Averaging_of_Symmetric_Positive-Definite_Tensors)
        """
        sqrt_q = q ** 0.5
        return sqrt_q * Quaternion.exp(eta) * sqrt_q

    @classmethod
    def log_map(cls, q, p):
        """Quaternion logarithm map.

        Find the logarithm map on the quaternion Riemannian manifold.

        Params:
             q: the base point at which the logarithm is computed, i.e.
                a Quaternion object
             p: the argument of the quaternion map, a Quaternion object

        Returns:
            A tangent vector having the length and direction given by the
            geodesic joining q and p.
        """
        return Quaternion.log(q.inverse * p)

    @classmethod
    def sym_log_map(cls, q, p):
        """Quaternion symmetrized logarithm map.

        Find the symmetrized logarithm map on the quaternion Riemannian manifold.

        Params:
             q: the base point at which the logarithm is computed, i.e.
                a Quaternion object
             p: the argument of the quaternion map, a Quaternion object

        Returns:
            A tangent vector corresponding to the symmetrized geodesic curve formulation.

        Note:
            Information on the symmetrized formulations given in [Source](https://www.researchgate.net/publication/267191489_Riemannian_L_p_Averaging_on_Lie_Group_of_Nonzero_Quaternions).
        """
        inv_sqrt_q = (q ** (-0.5))
        return Quaternion.log(inv_sqrt_q * p * inv_sqrt_q)

    @classmethod
    def absolute_distance(cls, q0, q1):
        """Quaternion absolute distance.

        Find the distance between two quaternions accounting for the sign ambiguity.

        Params:
            q0: the first quaternion
            q1: the second quaternion

        Returns:
           A positive scalar corresponding to the chord of the shortest path/arc that
           connects q0 to q1.

        Note:
           This function does not measure the distance on the hypersphere, but
           it takes into account the fact that q and -q encode the same rotation.
           It is thus a good indicator for rotation similarities.
        """
        q0_minus_q1 = q0 - q1
        q0_plus_q1  = q0 + q1
        d_minus = q0_minus_q1.norm
        d_plus  = q0_plus_q1.norm
        if d_minus < d_plus:
            return d_minus
        else:
            return d_plus

    @classmethod
    def distance(cls, q0, q1):
        """Quaternion intrinsic distance.

        Find the intrinsic geodesic distance between q0 and q1.

        Params:
            q0: the first quaternion
            q1: the second quaternion

        Returns:
           A positive amount corresponding to the length of the geodesic arc
           connecting q0 to q1.

        Note:
           Although the q0^(-1)*q1 != q1^(-1)*q0, the length of the path joining
           them is given by the logarithm of those product quaternions, the norm
           of which is the same.
        """
        q = Quaternion.log_map(q0, q1)
        return q.norm     
        
    @property
    def conjugate(self):
        """Quaternion conjugate, encapsulated in a new instance.

        For a unit quaternion, this is the same as the inverse.

        Returns:
            A new Quaternion object clone with its vector part negated
        """
        return self.Conjugate()

    @property
    def inverse(self):
        """Inverse of the quaternion object, encapsulated in a new instance.

        For a unit quaternion, this is the inverse rotation, i.e. when combined with the original rotation, will result in the null rotation.

        Returns:
            A new Quaternion object representing the inverse of this object
        """
        ss = self.norm()
        if ss > 0:
            self = self.muls(1/ss)
            return self.Conjugate()  #.__class__(array=(self.Conjugate() / ss))
        else:
            raise ZeroDivisionError("a zero quaternion (0 + 0i + 0j + 0k) cannot be inverted")

    @property
    def norm(self):
        """L2 norm of the quaternion 4-vector.

        This should be 1.0 for a unit quaternion (versor)
        Slow but accurate. If speed is a concern, consider using _fast_normalise() instead

        Returns:
            A scalar real number representing the square root of the sum of the squares of the elements of the quaternion.
        """
        return abs((self * self.Conjugate())[0])**0.5

    @property
    def magnitude(self):
        return self.norm
    
    @property
    def elements(self):
        """ Return all the elements of the quaternion object.

        Returns:
            A numpy 4-array of floats. NOT guaranteed to be a unit vector
        """
        return self.q    


    def _validate_number_sequence(self, seq, n):
        """Validate a sequence to be of a certain length and ensure it's a numpy array of floats.

        Raises:
            ValueError: Invalid length or non-numeric value
        """
        if seq is None:
            return np.zeros(n)
        if len(seq) == n:
            try:
                l = [float(e) for e in seq]
            except ValueError:
                raise ValueError("One or more elements in sequence <{!r}> cannot be interpreted as a real number".format(seq))
            else:
                return np.asarray(l)
        elif len(seq) == 0:
            return np.zeros(n)
        else:
            raise ValueError("Unexpected number of elements in sequence. Got: {}, Expected: {}.".format(len(seq), n))


    def __str__(self):
        if isinstance(self.q, list):
            res = ' + '.join(filter(None, [("%.7f" % x).rstrip("0").rstrip(".")+(["",self._base[i]][i>0]) if abs(x) > 0.000001 else None for i,x in enumerate(self)]))
        else:  # Assume array-like, redirect str conversion
            res = str(self.q)
        if (res == ''):
            return "0"
        return res

    def __getitem__(self, key):
        return self.q[key]

    def __setitem__(self, key, value):
        self.q[key] = value
        
    def __len__(self):
        return len(self.q)

    def __invert__(a):
        """Quaternion.Reverse
        
        Reverse the order of the basis blades.
        """
        res = a.q.copy()
        res[0]=a[0]
        res[1]=a[1]
        res[2]=a[2]
        res[3]=-a[3]
        return Quaternion.fromarray(res)

    def Dual(a):
        """Quaternion.Dual
        
        Poincare duality operator.
        """
        res = a.q.copy()
        res[0]=-a[3]
        res[1]=-a[2]
        res[2]=a[1]
        res[3]=a[0]
        return Quaternion.fromarray(res)

    def Conjugate(a):
        """Quaternion.Conjugate
        
        Clifford Conjugation
        """
        res = a.q.copy()
        res[0]=a[0]
        res[1]=-a[1]
        res[2]=-a[2]
        res[3]=-a[3]
        return Quaternion.fromarray(res)

    def Involute(a):
        """Quaternion.Involute
        
        Main involution
        """
        res = a.q.copy()
        res[0]=a[0]
        res[1]=-a[1]
        res[2]=-a[2]
        res[3]=a[3]
        return Quaternion.fromarray(res)

    def __mul__(a,b):
        """Quaternion.Mul
        
        The geometric product.
        """
        if type(b) in (int, float):
            return a.muls(b)
        res = a.q.copy()
        res[0]=b[0]*a[0]-b[1]*a[1]-b[2]*a[2]-b[3]*a[3]
        res[1]=b[1]*a[0]+b[0]*a[1]+b[3]*a[2]-b[2]*a[3]
        res[2]=b[2]*a[0]-b[3]*a[1]+b[0]*a[2]+b[1]*a[3]
        res[3]=b[3]*a[0]+b[2]*a[1]-b[1]*a[2]+b[0]*a[3]
        return Quaternion.fromarray(res)
    __rmul__=__mul__

    def __xor__(a,b):
        res = a.q.copy()
        res[0]=b[0]*a[0]
        res[1]=b[1]*a[0]+b[0]*a[1]
        res[2]=b[2]*a[0]+b[0]*a[2]
        res[3]=b[3]*a[0]+b[2]*a[1]-b[1]*a[2]+b[0]*a[3]
        return Quaternion.fromarray(res)


    def __and__(a,b):
        res = a.q.copy()
        res[3]=1*(a[3]*b[3])
        res[2]=-1*(a[2]*-1*b[3]+a[3]*b[2]*-1)
        res[1]=1*(a[1]*b[3]+a[3]*b[1])
        res[0]=1*(a[0]*b[3]+a[1]*b[2]*-1-a[2]*-1*b[1]+a[3]*b[0])
        return Quaternion.fromarray(res)


    def __or__(a,b):
        res = a.q.copy()
        res[0]=b[0]*a[0]-b[1]*a[1]-b[2]*a[2]-b[3]*a[3]
        res[1]=b[1]*a[0]+b[0]*a[1]+b[3]*a[2]-b[2]*a[3]
        res[2]=b[2]*a[0]-b[3]*a[1]+b[0]*a[2]+b[1]*a[3]
        res[3]=b[3]*a[0]+b[0]*a[3]
        return Quaternion.fromarray(res)


    def __add__(a,b):
        """Quaternion.Add
        
        Multivector addition
        """
        if type(b) in (int, float):
            return a.adds(b)
        res = a.q.copy()
        res[0] = a[0]+b[0]
        res[1] = a[1]+b[1]
        res[2] = a[2]+b[2]
        res[3] = a[3]+b[3]
        return Quaternion.fromarray(res)
    __radd__=__add__

    def __sub__(a,b):
        """Quaternion.Sub
        
        Multivector subtraction
        """
        if type(b) in (int, float):
            return a.subs(b)
        res = a.q.copy()
        res[0] = a[0]-b[0]
        res[1] = a[1]-b[1]
        res[2] = a[2]-b[2]
        res[3] = a[3]-b[3]
        return Quaternion.fromarray(res)

    def __rsub__(a,b):
        """Quaternion.Sub
                
        Multivector subtraction
        """
        return b + -1 * a


    def smul(a,b):
        res = a.q.copy()
        res[0] = a*b[0]
        res[1] = a*b[1]
        res[2] = a*b[2]
        res[3] = a*b[3]
        return Quaternion.fromarray(res)


    def muls(a,b):
        res = a.q.copy()
        res[0] = a[0]*b
        res[1] = a[1]*b
        res[2] = a[2]*b
        res[3] = a[3]*b
        return Quaternion.fromarray(res)


    def sadd(a,b):
        res = a.q.copy()
        res[0] = a+b[0]
        res[1] = b[1]
        res[2] = b[2]
        res[3] = b[3]
        return Quaternion.fromarray(res)


    def adds(a,b):
        res = a.q.copy()
        res[0] = a[0]+b
        res[1] = a[1]
        res[2] = a[2]
        res[3] = a[3]
        return Quaternion.fromarray(res)


    def ssub(a,b):
        res = a.q.copy()
        res[0] = a-b[0]
        res[1] = -b[1]
        res[2] = -b[2]
        res[3] = -b[3]
        return Quaternion.fromarray(res)


    def subs(a,b):
        res = a.q.copy()
        res[0] = a[0]-b
        res[1] = a[1]
        res[2] = a[2]
        res[3] = a[3]
        return Quaternion.fromarray(res)

    def norm(a):
        return abs((a * a.Conjugate())[0])**0.5
        
    def inorm(a):
        return a.Dual().norm()
        
    def normalized(a):
        return a * (1 / a.norm())

    # Division
    def __div__(self, other):
        if isinstance(other, Quaternion):
            if other == self.__class__(0.0):
                raise ZeroDivisionError("Quaternion divisor must be non-zero")
            return self * other.inverse
        else:
            return self.__div__(self.__class__(other))

    def __idiv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        return self.__class__(other) * self.inverse

    def __truediv__(self, other):
        return self.__div__(other)
    
    # Exponentiation
    def __pow__(self, exponent):
        # source: https://en.wikipedia.org/wiki/Quaternion#Exponential.2C_logarithm.2C_and_power
        exponent    = float(exponent) # Explicitly reject non-real exponents
        norm        = self.norm
        if norm > 0.0:
            try:
                n, theta = self.polar_decomposition
            except ZeroDivisionError:
                # quaternion is a real number (no vector or imaginary part)
                return Quaternion(scalar=self.scalar ** exponent)
            return (self.norm ** exponent) * Quaternion(scalar=cos(exponent * theta), vector=(n * sin(exponent * theta)))
        return Quaternion(self)

    def __ipow__(self, other):
        return self ** other

    def __rpow__(self, other):
        return other ** float(self)    

if __name__ == '__main__':

    e1  = Quaternion(1.0, [1,0,0])
    e2  = Quaternion(1.0, [0,1,0])
    e12 = Quaternion(1.0, [0,0,1])
    q1  = Quaternion(1.0, [1,1,0])
    
    print("e1*e1         :", str(e1*e1))
    print("pss           :", str(e12))
    print("pss*pss       :", str(e12*e12))
    print("e1/q1         :", str(e1/q1))

